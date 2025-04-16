import multiprocessing

# Bind to Render's port
bind = "0.0.0.0:8000"

# Use gevent for async workers
worker_class = "gevent"

# Optimize worker count for Render's resource limits
workers = multiprocessing.cpu_count() * 2 + 1  # Typically 2-4 workers on Render Starter
threads = 2  # Limited threads with gevent

# Timeout settings to prevent hanging requests
timeout = 25  # Render's default is ~30s, so keep this lower
keepalive = 5

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Preload app to reduce memory usage
preload_app = True
