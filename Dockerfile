FROM python:3.13-slim
WORKDIR /app

# 1. Instalar dependencias de sistema (Gobject, Cairo, Introspection, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    pkg-config \
    # Dependencias de GObject/Cairo/XML (lxml):
    libxml2-dev \
    libxslt-dev \
    libcairo2-dev \
    libglib2.0-dev \
    libgirepository1.0-dev \
    # ESTE ES EL NUEVO PAQUETE CRÍTICO (GObject-Introspection para 'girepository-2.0')
    # Este paquete proporciona las herramientas que Meson está buscando
    gobject-introspection && \
    rm -rf /var/lib/apt/lists/*

# 2. Instalar los requerimientos de Python (numpy, pygobject, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --default-timeout=3600 \
    -r requirements.txt

# 3. Instalación de PyTorch (solo CPU) - Mantenlo si es necesario
RUN pip install --no-cache-dir \
    --default-timeout=3600 \
    "torch==2.9.0" \
    --index-url https://download.pytorch.org/whl/cpu

# 4. Limpiar herramientas de compilación para reducir el tamaño final
RUN apt-get purge -y gcc build-essential pkg-config libxml2-dev libxslt-dev \
    libcairo2-dev libgirepository1.0-dev libglib2.0-dev gobject-introspection && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 5. Código de la aplicación y ejecución
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]