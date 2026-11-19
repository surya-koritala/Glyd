$ErrorActionPreference = 'Stop'
# Set a normal console disposition before starting the isolated Rust test. CI's
# shell may inherit CTRL_C suppression; installing a handler does not reset it.
Add-Type @'
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
public static class ConsoleTest {
    [DllImport("kernel32.dll", SetLastError=true)] private static extern bool SetConsoleCtrlHandler(IntPtr handler, bool add);
    public static int Run(string executable) {
        if (!SetConsoleCtrlHandler(IntPtr.Zero, false)) throw new Win32Exception(Marshal.GetLastWin32Error());
        var start = new ProcessStartInfo(executable, "--exact windows_console_events_preserve_ordered_cleanup --nocapture");
        start.UseShellExecute = false;
        start.EnvironmentVariables["STARLINGS_TEST_CONSOLE_READY"] = "1";
        using (var child = Process.Start(start)) {
            if (!child.WaitForExit(60000)) {
                child.Kill();
                child.WaitForExit();
                throw new TimeoutException("Isolated console test exceeded deadline");
            }
            return child.ExitCode;
        }
    }
}
'@
exit [ConsoleTest]::Run($env:STARLINGS_TEST_EXECUTABLE)
