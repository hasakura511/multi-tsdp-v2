from bigspool.celery import app


@app.task(bind=True)
def get_quotes(self):
    pass
