# Utiliser une image de base Python 3.9-slim
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le code source dans le répertoire de travail
COPY . /app/

# Supprimer l'environnement virtuel existant s'il y en a un
RUN rm -rf /app/venv

# Créer et activer un nouvel environnement virtuel
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Exposer le port nécessaire pour l'application
EXPOSE 8085

# Commande pour exécuter l'application
CMD ["python", "app.py"]
