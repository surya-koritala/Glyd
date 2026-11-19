//! Explicit local addresses only. Discovery never supplies a trusted route.
use crate::{Error, Result};
use serde::Serialize;
use socket2::{Domain, Protocol, Socket, Type};
use std::{
    net::{IpAddr, SocketAddr, TcpStream},
    time::Duration,
};

pub const MAX_ROUTES: usize = 4;
pub const POLICY: &str =
    "explicit-local-unicast; no DNS, auto-enrollment, router changes or public destinations";

pub fn local_address(address: SocketAddr, allow_zero_port: bool) -> bool {
    if !allow_zero_port && address.port() == 0 {
        return false;
    }
    match address.ip() {
        IpAddr::V4(ip) => ip.is_loopback() || ip.is_private() || ip.is_link_local(),
        IpAddr::V6(ip) => {
            if ip.is_loopback() || ip.is_unique_local() {
                return true;
            }
            ip.is_unicast_link_local() && matches!(address, SocketAddr::V6(a) if a.scope_id() != 0)
        }
    }
}

pub fn validate_routes(routes: &[SocketAddr], source: Option<SocketAddr>) -> Result<()> {
    if routes.is_empty()
        || routes.len() > MAX_ROUTES
        || routes
            .iter()
            .enumerate()
            .any(|(i, a)| !local_address(*a, false) || routes[..i].contains(a))
        || source.is_some_and(|a| {
            !local_address(a, true)
                || a.port() != 0
                || routes.iter().any(|r| r.is_ipv4() != a.is_ipv4())
        })
    {
        return Err(Error::Invalid);
    }
    Ok(())
}

pub(crate) fn connect(
    address: SocketAddr,
    source: Option<SocketAddr>,
    timeout: Duration,
) -> Result<TcpStream> {
    validate_routes(&[address], source)?;
    let socket = Socket::new(
        Domain::for_address(address),
        Type::STREAM,
        Some(Protocol::TCP),
    )?;
    if let Some(source) = source {
        socket.bind(&source.into())?;
    }
    socket.connect_timeout(&address.into(), timeout)?;
    Ok(socket.into())
}

#[derive(Clone, Debug, Serialize)]
pub struct Interface {
    pub name: String,
    pub index: Option<u32>,
    pub address: IpAddr,
}
pub fn interfaces() -> Result<Vec<Interface>> {
    let mut items: Vec<_> = if_addrs::get_if_addrs()?
        .into_iter()
        .map(|i| Interface {
            address: i.ip(),
            name: i.name,
            index: i.index,
        })
        .collect();
    items.sort_by(|a, b| (&a.name, a.address).cmp(&(&b.name, b.address)));
    if items.len() > 32 || items.iter().any(|i| i.name.len() > 128) {
        return Err(Error::Capacity);
    }
    Ok(items)
}

pub(crate) fn selected_interface(address: SocketAddr) -> Option<Interface> {
    #[cfg(test)]
    assert!(
        INTERFACE_LOOKUP_ALLOWED.with(|allowed| allowed.get()),
        "interface enumeration is forbidden on this control path"
    );
    interfaces().ok()?.into_iter().find(|i| {
        i.address == address.ip()
            && match address {
                SocketAddr::V6(a) if a.scope_id() != 0 => i.index == Some(a.scope_id()),
                _ => true,
            }
    })
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static INTERFACE_LOOKUP_ALLOWED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn address_policy_refuses_public_wildcard_multicast_and_ambiguous_scope() {
        for a in [
            "0.0.0.0:1",
            "8.8.8.8:53",
            "224.0.0.251:5353",
            "255.255.255.255:1",
            "[::]:1",
            "[ff02::fb]:5353",
            "[fe80::1]:1",
            "[::ffff:127.0.0.1]:1",
            "127.0.0.1:0",
        ] {
            assert!(!local_address(a.parse().unwrap(), false), "{a}");
        }
        for a in [
            "127.0.0.1:1",
            "192.168.1.2:1",
            "10.0.0.1:1",
            "172.16.0.1:1",
            "169.254.1.1:1",
            "[::1]:1",
            "[fd00::1]:1",
            "[fe80::1%2]:1",
        ] {
            assert!(local_address(a.parse().unwrap(), false), "{a}");
        }
        let a = "127.0.0.1:1".parse().unwrap();
        assert!(validate_routes(&[a, a], None).is_err());
        assert!(validate_routes(&[], None).is_err());
        assert!(validate_routes(&[a], Some("[::1]:0".parse().unwrap())).is_err());
        assert!(
            interfaces()
                .unwrap()
                .iter()
                .any(|i| i.address.is_loopback())
        );
    }
}
