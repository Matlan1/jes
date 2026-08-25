import sys
import os
import time
import datetime
import traceback
import platform
import numpy as np
import ctypes
from ctypes import wintypes

def get_system_memory_info():
    """Returns (process_ram_mb, process_peak_mb, total_ram_gb, avail_ram_gb, mem_load_pct,
    process_commit_mb) using Win32 API."""
    rss_mb, peak_mb, total_gb, avail_gb, load_pct, commit_mb = -1.0, -1.0, -1.0, -1.0, -1, -1.0
    
    # Process memory
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
            ]
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            rss_mb = counters.WorkingSetSize / (1024 * 1024)
            peak_mb = counters.PeakWorkingSetSize / (1024 * 1024)
            commit_mb = counters.PagefileUsage / (1024 * 1024)
    except Exception:
        pass

    # System-wide memory
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', wintypes.DWORD),
                ('dwMemoryLoad', wintypes.DWORD),
                ('ullTotalPhys', ctypes.c_uint64),
                ('ullAvailPhys', ctypes.c_uint64),
                ('ullTotalPageFile', ctypes.c_uint64),
                ('ullAvailPageFile', ctypes.c_uint64),
                ('ullTotalVirtual', ctypes.c_uint64),
                ('ullAvailVirtual', ctypes.c_uint64),
                ('ullAvailExtendedVirtual', ctypes.c_uint64),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_gb = stat.ullTotalPhys / (1024 ** 3)
            avail_gb = stat.ullAvailPhys / (1024 ** 3)
            load_pct = stat.dwMemoryLoad
    except Exception:
        pass

    return rss_mb, peak_mb, total_gb, avail_gb, load_pct, commit_mb


def log_crash(exc_type, exc_value, exc_tb, sim=None, ui=None):
    """Writes a comprehensive diagnostic report and emergency state dump to disk."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    current_gen = len(sim.percentiles) if (sim and hasattr(sim, 'percentiles')) else "Unknown"
    report_filename = os.path.join(reports_dir, f"crash_report_gen_{current_gen}_{timestamp}.txt")
    emergency_dump_filename = os.path.join(reports_dir, f"emergency_save_gen_{current_gen}_{timestamp}.npz")
    
    rss_mb, peak_mb, total_gb, avail_gb, load_pct, commit_mb = get_system_memory_info()
    formatted_tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    
    lines = []
    lines.append("=" * 80)
    lines.append("        JELLY EVOLUTION SIMULATOR - FAULT ANALYSIS & CRASH REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Exception Type:  {exc_type.__name__ if hasattr(exc_type, '__name__') else str(exc_type)}")
    lines.append(f"Exception Value: {exc_value}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("1. ENVIRONMENT & SYSTEM DIAGNOSTICS")
    lines.append("-" * 80)
    lines.append(f"OS / Platform:   {platform.platform()} ({platform.machine()})")
    lines.append(f"Python Version:  {sys.version.split()[0]} ({platform.architecture()[0]})")
    lines.append(f"Process RAM RSS: {rss_mb:.2f} MB (Peak: {peak_mb:.2f} MB, Commit: {commit_mb:.2f} MB)" if rss_mb >= 0 else "Process RAM RSS: N/A")
    lines.append(f"System RAM:      {total_gb:.2f} GB total, {avail_gb:.2f} GB free ({load_pct}% used)" if total_gb >= 0 else "System RAM: N/A")
    lines.append(f"NumPy Version:   {np.__version__}")
    
    try:
        from utils import HAS_NUMBA
        lines.append(f"Numba JIT:       {'Enabled (Multi-Core Parallel)' if HAS_NUMBA else 'Disabled (Single-Core)'}")
    except Exception:
        pass
    lines.append("")

    lines.append("-" * 80)
    lines.append("2. SIMULATION STATE AT MOMENT OF CRASH")
    lines.append("-" * 80)
    if sim is not None:
        try:
            gen_count = len(sim.percentiles)
            lines.append(f"Current Generation:   {gen_count}")
            lines.append(f"Population Size (N):  {sim.c_count}")
            lines.append(f"Total Species Born:   {sim.species_count if hasattr(sim, 'species_count') else len(sim.species_info)}")
            
            if hasattr(sim, 'species_pops') and len(sim.species_pops) > 0:
                active_sps = len(sim.species_pops[-1])
                lines.append(f"Active Species:       {active_sps}")
            if hasattr(sim, 'current_leader'):
                lines.append(f"Dominant Species ID:  {sim.current_leader} (Tenure: {sim.leader_tenure} gens)")
            if hasattr(sim, 'last_gen_run_time'):
                lines.append(f"Last Gen Runtime:     {sim.last_gen_run_time * 1000:.1f} ms")
            
            # Recent Fitness Trajectory (Last 15 generations)
            if hasattr(sim, 'percentiles') and len(sim.percentiles) > 0:
                u = getattr(sim, 'UNITS_PER_METER', 0.05) * 100 # to cm
                lines.append("\nRecent Generation Fitness Trajectory (Last 15 gens):")
                lines.append(f"{'Gen':>8} | {'Best (cm)':>12} | {'Median (cm)':>12} | {'Worst (cm)':>12}")
                lines.append("-" * 52)
                start_g = max(0, len(sim.percentiles) - 15)
                for g in range(start_g, len(sim.percentiles)):
                    p_row = sim.percentiles[g]
                    best = p_row[0] * u
                    med = (p_row[50] if len(p_row) > 50 else p_row[len(p_row)//2]) * u
                    worst = p_row[-1] * u
                    lines.append(f"{g+1:8d} | {best:12.2f} | {med:12.2f} | {worst:12.2f}")
        except Exception as sim_err:
            lines.append(f"Error extracting simulation state: {sim_err}")
    else:
        lines.append("Simulation instance was not initialized.")
    lines.append("")

    lines.append("-" * 80)
    lines.append("3. FULL PYTHON TRACEBACK")
    lines.append("-" * 80)
    lines.append(formatted_tb)
    lines.append("=" * 80)
    
    # Write report file
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as io_err:
        print(f"Failed to write crash report to disk: {io_err}", file=sys.stderr)

    # Save emergency checkpoint
    saved_checkpoint = False
    if sim is not None and hasattr(sim, 'creatures') and sim.creatures is not None and len(sim.creatures) > 0:
        try:
            last_gen_creatures = sim.creatures[-1]
            dna_matrix = np.array([c.dna for c in last_gen_creatures if c is not None])
            species_arr = np.array([c.species for c in last_gen_creatures if c is not None])
            fitness_arr = np.array([c.fitness if c.fitness is not None else 0.0 for c in last_gen_creatures if c is not None])
            percentiles_arr = np.array(sim.percentiles) if len(sim.percentiles) > 0 else np.zeros((0,))
            
            np.savez_compressed(
                emergency_dump_filename,
                dna=dna_matrix,
                species=species_arr,
                fitness=fitness_arr,
                percentiles=percentiles_arr,
                gen=len(sim.percentiles)
            )
            saved_checkpoint = True
        except Exception as dump_err:
            print(f"Failed to write emergency checkpoint: {dump_err!r}", file=sys.stderr)

    # Prominent Console Output
    print("\n" + "!" * 80)
    print("                      CRITICAL SIMULATION ERROR CAUGHT")
    print("!" * 80)
    print(f"An unexpected error occurred: {exc_type.__name__ if hasattr(exc_type, '__name__') else str(exc_type)}: {exc_value}")
    print(f"\n[+] Full Crash Diagnostics written to:\n    -> {report_filename}")
    if saved_checkpoint:
        print(f"[+] Emergency Universe State saved to:\n    -> {emergency_dump_filename}")
    print("!" * 80 + "\n")


def setup_global_exception_handler(sim=None, ui=None):
    """Installs sys.excepthook to intercept any unhandled exceptions."""
    def excepthook_handler(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            # Normal user exit via Ctrl+C
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log_crash(exc_type, exc_value, exc_tb, sim=sim, ui=ui)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = excepthook_handler
