# Direct routes, discovery and topology

GLY-169 adds explicit local routes and optional DNS-SD hints. Enrollment, certificate
pins and registry authority remain the source of trust. Discovery never changes
configuration, downloads a model, enrolls a node or opens an inference route.

## Direct addresses and multiple interfaces

Use one to four concrete listener addresses. All listeners share one admission
budget and the same coordinator identity. The default remains loopback. These
private addresses are examples; substitute addresses actually assigned to the host:

```sh
starlings --state-dir /absolute/coordinator cluster init --listen 192.168.10.10:9741 --listen 10.20.0.10:9741
starlings --state-dir /absolute/coordinator daemon
starlings --state-dir /absolute/coordinator cluster topology
starlings --state-dir /absolute/coordinator cluster invite worker /absolute/coordinator/invite.json --via 10.20.0.10:9741
```

An invitation uses the first listener unless explicitly selected with --via. That
address must be an active listener. Transfer the private invitation and compare
the device fingerprint using the [enrollment procedure](daemon.md). A joining
device can explicitly select another address of the same pinned coordinator:

```sh
starlings --state-dir /absolute/worker cluster join /absolute/private/invite.json --via 192.168.10.10:9741
# With the worker daemon stopped:
starlings --state-dir /absolute/worker cluster routes 192.168.10.10:9741 10.20.0.10:9741
starlings --state-dir /absolute/worker daemon
starlings --state-dir /absolute/worker cluster topology
```

Routes replaces the complete list in protected config. There are at most four
unique addresses. With no override the invitation's address is used. The optional
--source IP:0 binds outbound registry connections to a local address before connect;
all destinations must then have the same IP family. Without it, OS routing chooses
the source. Enrollment uses OS routing; --via chooses its destination. There is no
DNS resolution or implicit fallback to unconfigured addresses.

One address is tried per heartbeat, with the existing three-second total exchange
deadline and two-to-three-second delay. Success sticks to that address; failure
advances to the next address on the next heartbeat. All addresses must authenticate
the same enrolled coordinator. Up to four unreachable routes can delay reconnection;
they cannot extend six-second authority/resource expiry or existing work leases.
This is alternate network paths to one coordinator, not coordinator failover.

Policy permits IPv4 loopback, RFC 1918 private and link-local addresses, plus IPv6
loopback, unique-local and scoped link-local addresses. IPv6 link-local requires a
numeric scope index, such as [fe80::1%4]:9741; --source likewise takes a scoped
socket address with port zero. Public/global addresses, wildcard destinations,
multicast, IPv4-mapped IPv6 and zero destination ports are refused. Listener port
zero is permitted for tests; normal deployments need a stable chosen port. A LAN
using globally assigned IPv6 is outside this initial policy. No firewall, router,
NAT mapping or Internet publication is configured.

## Optional mDNS hints

mDNS is disabled unless explicitly invoked. Initial hints use one selected local
IPv4 interface; direct TLS routes independently support IPv6. The coordinator must
already listen on the selected address:

```sh
starlings --state-dir /absolute/coordinator daemon --mdns 192.168.10.10
starlings cluster discover --interface 192.168.10.20
```

The service type is _starlings._tcp.local. Advertisement contains a public node
identifier, local hostname, listener address/port and only the TXT marker v=1.
There is no cluster secret, certificate, token, invitation, model inventory or
automatic enrollment. Do not infer identity from the public node name.

A scan lasts at most two seconds, processes at most 64 events and displays at most
16 unique filtered endpoints. Output is structured JSON labelled UNTRUSTED_HINTS
with trusted=false. Unsupported versions, unexpected TXT fields, nonlocal addresses
and zero ports are discarded. Arbitrary service names and TXT contents are never
printed. No hint is persisted or automatically contacted. Verify a separately
obtained invitation and explicitly select an address; TLS still rejects a wrong
coordinator even when that address was selected manually.

mdns-sd owns DNS parsing, protocol timing and native multicast sockets. All
interfaces are disabled before enabling the selected address; unsolicited caching
and periodic interface auto-detection are disabled. Scans stop browsing and shut
down their daemon. Advertisement lifetime ends with the coordinator process.
The library has internal DNS/cache allocations; the scan/result bounds are not
a hard RSS or hostile-LAN flood guarantee. It is optional and isolated from the
TLS trust path. OS multicast restrictions can prevent hints; direct enrollment
and registry work without multicast. Interface/address changes require restart.

## Records and diagnostics

Existing strict config.json retains version, role, cluster, node, listen and
cache_dir. It adds additional_listen (default []), routes (default []) and source
(default null), preserving existing profiles. Duplicate, excessive or policy-invalid
addresses fail before binding. Runtime changes require stopping the profile;
the existing process lock prevents concurrent edits. Existing local admin invite
adds optional via; topology is a new authenticated local command.

RegistryClient::configure_routes accepts only explicit locally approved addresses.
RegistryFence::topology returns one RouteSummary per configured address:

| Field | Meaning |
| --- | --- |
| destination | Configured numeric coordinator address |
| local_address | Actual source socket address from the attempted connection, or null |
| selected_interface | Current OS interface name, index and address matching the recorded source, or null if unknown |
| authenticated | Latest exchange succeeded and remains within registry freshness |
| age_ms | Receiver-observed age of the attempt, or null if never tried/clock invalid |
| outcome | untried, authenticated, expired, or a redacted failure code |

Interface labels are resolved only during explicit topology observation. Heartbeat
completion records the actual source socket without enumerating OS interfaces.
Topology copies the bounded route records and releases the registry fence before
lookup; age and authentication freshness are evaluated after all lookups finish.
Interface labels do not grant trust or prove that an interface name is unchanged
since the recorded attempt. An OS inventory query may delay the diagnostic reply,
but does not hold the worker's live fence mutex.

Topology reports local interface inventory (at most 32 entries; names at most
128 bytes), active listeners, mDNS interface and explicit network policy. Its
direction is this_node_to_coordinator. Reverse and worker-to-worker reachability
are explicitly unknown; a reply on an outbound connection does not prove a new
inbound connection would work. Authenticated reachability is not inference
schedulability, latency/bandwidth measurement, model authorization or a resource
probe. Network diagnostics stay behind local admin authentication.

## Validation boundary

Native process tests exercise two listeners, explicit invitation/alternate join
addresses, failed-route fallback, source binding, authenticated topology and
configuration locking with multicast off. TLS tests reject a different coordinator,
then recover on the pinned address and expire old route observations. Hint filtering
uses the real library's resolved-service type. A separate macOS loopback smoke run
verified real advertisement, discovery and shutdown. CI does not depend on
multicast being allowed by hosted runners.

Physical Ethernet/Wi-Fi/VPN asymmetry, IPv6 scope selection on real interfaces,
cross-machine multicast, Pi networking and hostile multicast flood qualification
remain unmeasured. No inference, throughput or speed claim is made.

Sources: [mdns-sd 0.21.5](https://docs.rs/mdns-sd/0.21.5/mdns_sd/),
[socket2 0.6.5](https://docs.rs/socket2/0.6.5/socket2/struct.Socket.html),
[if-addrs 0.15.0](https://docs.rs/if-addrs/0.15.0/if_addrs/).
