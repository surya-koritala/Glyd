# GLY-198 inactive per-user descriptors

Issue: [GLY-198](https://linear.app/getglyd/issue/GLY-198).
Base: PR100 merge18fdb20ce055ce45ea28499726af449d03b5e09a.

The CLI reuses existing state-path normalization, strict bounded config parsing
and private-fs capabilities. It creates a new private review directory and one
exclusive file. Profile config is read-only; no state lock, identity read, enrollment,
daemon/admin call, network, installer or service-manager invocation occurs.
Native Linux/macOS formats use the fixed current-executable/profile/daemon argv,
restrictive umask and no automatic restart. macOS uses native plutil rather than
a new plist dependency. [Packaging](../packaging.md) links the primary grammar
and OS references and records activation/installation limits.

One bounded file-only check exercises quoting, literal dollar/percent/Unicode,
invalid/control/overlong paths, create-new behavior, private modes, symlink output
and parent refusal, unchanged profile bytes and no cache/state-lock creation.
macOS's native plutil reads the generated XML back into JSON and checks the exact
four argv elements, label, mask and disabled auto-start/restart. Linux renderer
assertions are syntax expectations, not a running-manager qualification.

Initial fixture attempts were refused because the default temporary directory
was not a private export parent. The fixture now creates an owned private child
using the existing capability API. Dollar, percent, quote, backslash and Unicode
remain in the actual profile name; no path or permission guard was relaxed.
Local execution is limited to this synthetic check, formatting and denied-warning
compilation/lint with two jobs. No socket, service or model test runs on the Mac.

Local check passes0.06s after correcting the fixture's private parent; default
and all-feature warning-denied workspace/all-target lint checks pass5.79s/3.22s.
Formatting and diff checks pass. Separate complexity review found no redundant
framework/dependency/installer; correctness/security review verified private
exclusive creation, strict bounded inputs, fixed direct argv, no manager/network
calls, no secret copying and unchanged launch/approval policy. Linux native
manager qualification and all platform CI remain required; no review approval is
inferred from a local pass.

PR100's exact source passed all seven checks, but its separate automatic main
CI38932851682 failed Windows backend/aggregate during token creation; five checks
passed. Raw log SHA2563de16ceffdfa308acbdcf50ad9f9c956cb7fd33bf75657786e1c9ee6e01fefa7
is retained. This descriptor change does not claim to resolve that historical
initiator. No unchanged-source rerun or timing/durability weakening is introduced.

Installed binary stability, Linux manager validation, all target builds, WSL2,
login/logout, crash/stop/drain, upgrade/rollback/uninstall, native hardware/model
qualification, license/release decisions and independent review remain open.
Publication is not approved. GLY-198 remains In Progress.
