from django.conf import settings

# how many seconds before we give up
MAX_WAIT_SECONDS = settings.getattr('IB_MAX_WAIT_SECONDS', 30)