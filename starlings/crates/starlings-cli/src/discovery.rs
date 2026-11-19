//! Opt-in DNS-SD hints. This module has no invitation, key or enrollment access.
use crate::cluster::{Result, security};
use mdns_sd::{IfKind, ResolvedService, ServiceDaemon, ServiceEvent, ServiceInfo};
use starlings_protocol::NodeId;
use starlings_security::network;
use std::{
    collections::BTreeSet,
    net::{IpAddr, SocketAddr},
    time::{Duration, Instant},
};

const SERVICE: &str = "_starlings._tcp.local.";
const SCAN_TIME: Duration = Duration::from_secs(2);

fn failed(_: impl std::fmt::Display) -> String {
    "MDNS_UNAVAILABLE: use an explicit coordinator address".into()
}

pub struct Advertisement(ServiceDaemon);
impl Drop for Advertisement {
    fn drop(&mut self) {
        if let Ok(done) = self.0.shutdown() {
            let _ = done.recv_timeout(Duration::from_secs(1));
        }
    }
}
fn start(ip: IpAddr) -> Result<Advertisement> {
    // Initial hints use a selected IPv4 interface; direct TLS also supports scoped IPv6.
    if !ip.is_ipv4()
        || !network::local_address(SocketAddr::new(ip, 5353), false)
        || !network::interfaces()
            .map_err(security)?
            .iter()
            .any(|i| i.address == ip)
    {
        return Err(
            "MDNS_INTERFACE_REQUIRED: choose a local private or loopback IPv4 address".into(),
        );
    }
    let daemon = Advertisement(ServiceDaemon::new().map_err(failed)?);
    daemon.0.disable_interface(IfKind::All).map_err(failed)?;
    daemon.0.enable_interface(ip).map_err(failed)?;
    daemon.0.set_ip_check_interval(0).map_err(failed)?;
    daemon.0.accept_unsolicited(false).map_err(failed)?;
    Ok(daemon)
}
pub fn advertise(address: SocketAddr, node: NodeId) -> Result<Advertisement> {
    let daemon = start(address.ip())?;
    let service = ServiceInfo::new(
        SERVICE,
        &node.to_string(),
        &format!("{node}.local."),
        address.ip(),
        address.port(),
        &[("v", "1")][..],
    )
    .map_err(failed)?;
    daemon.0.register(service).map_err(failed)?;
    Ok(daemon)
}
fn hints(service: &ResolvedService) -> Vec<SocketAddr> {
    if !service.get_fullname().ends_with(&format!(".{SERVICE}"))
        || service.get_property_val_str("v") != Some("1")
        || service.get_properties().iter().count() != 1
    {
        return Vec::new();
    }
    service
        .get_addresses_v4()
        .into_iter()
        .map(|ip| SocketAddr::new(IpAddr::V4(ip), service.get_port()))
        .filter(|a| network::local_address(*a, false))
        .take(16)
        .collect()
}
pub fn browse(ip: IpAddr) -> Result<()> {
    let daemon = start(ip)?;
    let events = daemon.0.browse(SERVICE).map_err(failed)?;
    let deadline = Instant::now() + SCAN_TIME;
    let mut found = BTreeSet::new();
    // ponytail: two-second, 64-event/16-result scan; no persistent discovery cache.
    for _ in 0..64 {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            break;
        };
        let Ok(event) = events.recv_timeout(remaining) else {
            break;
        };
        if let ServiceEvent::ServiceResolved(service) = event {
            for address in hints(&service) {
                if found.len() < 16 {
                    found.insert(address);
                }
            }
        }
    }
    daemon.0.stop_browse(SERVICE).map_err(failed)?;
    println!(
        "UNTRUSTED_HINTS: {}",
        serde_json::json!({
            "endpoints": found, "interface": ip, "trusted": false,
            "action": "verify invitation fingerprint; explicitly select an address; no enrollment performed"
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hints_are_filtered_without_becoming_authority_or_exposing_txt() {
        let info = ServiceInfo::new(
            SERVICE,
            "untrusted",
            "untrusted.local.",
            "127.0.0.1,8.8.8.8",
            9741,
            &[("v", "1")][..],
        )
        .unwrap();
        let mut resolved = info.as_resolved_service();
        assert_eq!(hints(&resolved), vec!["127.0.0.1:9741".parse().unwrap()]);
        resolved.port = 0;
        assert!(hints(&resolved).is_empty());
        let info = ServiceInfo::new(
            SERVICE,
            "untrusted",
            "untrusted.local.",
            "127.0.0.1",
            9741,
            &[("v", "1"), ("token", "never-display")][..],
        )
        .unwrap();
        assert!(hints(&info.as_resolved_service()).is_empty());
        assert!(start("8.8.8.8".parse().unwrap()).is_err());
    }
}
