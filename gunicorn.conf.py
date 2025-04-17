# Number of worker processes
workers = 2

# Timeout for worker processing
timeout = 60

# Logging level
loglevel = "info"

# Bind to Render's dynamic port
bind = "0.0.0.0:$PORT"

# Worker class
worker_class = "sync"

# Restart workers after this many requests to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Log to stdout for Render
errorlog = "-"
accesslog = "-"

# Number of threads per worker
threads = 2
