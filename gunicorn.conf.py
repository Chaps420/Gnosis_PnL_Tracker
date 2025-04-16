# Bind to Render's port
bind = "0.0.0.0:8000"

# Use sync workers to avoid gevent issues
worker_class = "sync"

# Minimize workers for Render's 512 MB limit
workers = 1  # Single worker to conserve memory

# Single thread per worker
threads = 1

# Timeout settings for Render's ~30s limit
timeout = 25
keepalive = 5

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Disable preload to reduce initial memory usage
preload_app = False
