import multiprocessing

# Bind to Render's port
bind = "0.0.0.0:8000"

# Use gevent for async workers, fallback to sync if gevent fails
worker_class = "gevent"

# Conservative worker count for Render Starter...
