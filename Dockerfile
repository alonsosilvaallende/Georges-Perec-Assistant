FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN uv pip install -r requirements.txt

COPY --chown=user . /app
# Switch to the "user" user
USER user

CMD ["solara", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
