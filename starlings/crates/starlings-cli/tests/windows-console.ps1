$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$owner = (Get-Acl -LiteralPath $env:STARLINGS_TEST_STATE).GetOwner([Security.Principal.SecurityIdentifier]).Value
if ($owner -ne $identity.User.Value) { throw 'Profile owner does not match the explicit user SID' }
$capability = Get-Acl -LiteralPath ([IO.Path]::Combine($env:STARLINGS_TEST_STATE, 'admin.cap'))
if (-not $capability.AreAccessRulesProtected -or $capability.GetOwner([Security.Principal.SecurityIdentifier]).Value -ne $owner) {
    throw 'Capability ownership or DACL protection changed'
}
# The helper joins only the test daemon's private console; never the CI runner console.
Add-Type @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Threading;
public static class ConsoleSignal {
    private delegate bool Handler(uint signal);
    private static readonly Handler KeepHelperAlive = signal => true;
    [DllImport("kernel32.dll", SetLastError=true)] private static extern bool FreeConsole();
    [DllImport("kernel32.dll", SetLastError=true)] private static extern bool AttachConsole(uint process);
    [DllImport("kernel32.dll", SetLastError=true)] private static extern bool SetConsoleCtrlHandler(Handler handler, bool add);
    [DllImport("kernel32.dll", SetLastError=true)] private static extern bool GenerateConsoleCtrlEvent(uint signal, uint group);
    public static void Send(uint process, uint signal) {
        FreeConsole();
        if (!AttachConsole(process)) throw new Win32Exception(Marshal.GetLastWin32Error());
        if (!SetConsoleCtrlHandler(KeepHelperAlive, true)) throw new Win32Exception(Marshal.GetLastWin32Error());
        if (!GenerateConsoleCtrlEvent(signal, 0)) throw new Win32Exception(Marshal.GetLastWin32Error());
        Thread.Sleep(150);
        FreeConsole();
        GC.KeepAlive(KeepHelperAlive);
    }
}
'@
[ConsoleSignal]::Send([uint32]$env:STARLINGS_TEST_PID, [uint32]$env:STARLINGS_TEST_SIGNAL)
