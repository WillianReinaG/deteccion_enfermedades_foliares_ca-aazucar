#!/usr/bin/env python3
"""Genera PDF de arquitectura Docker / Docker Hub / despliegue SugarCane."""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUTPUT = Path(
    r"C:\Users\bebes\Documents\MIAA\3.SEMESTRE\3.proyecto_tres"
    r"\SugarCane_Arquitectura_Docker_y_Despliegue.pdf"
)
FONT_DIR = Path(r"C:\Windows\Fonts")


class DocPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")


def _w(pdf: DocPDF) -> float:
    return pdf.epw


def section_title(pdf: DocPDF, text: str) -> None:
    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(0, 80, 40)
    pdf.multi_cell(_w(pdf), 8, text)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def body(pdf: DocPDF, text: str) -> None:
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(_w(pdf), 6, text)
    pdf.ln(1)


def bullet(pdf: DocPDF, text: str) -> None:
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(_w(pdf), 6, f"  - {text}")


def code_block(pdf: DocPDF, text: str) -> None:
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(_w(pdf), 5, text)
    pdf.set_font("Arial", "", 11)


def build_pdf() -> None:
    pdf = DocPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_font("Arial", "", str(FONT_DIR / "arial.ttf"))
    pdf.add_font("Arial", "B", str(FONT_DIR / "arialbd.ttf"))
    pdf.add_font("Arial", "I", str(FONT_DIR / "ariali.ttf"))
    pdf.set_margins(20, 20, 20)

    # Portada
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(0, 100, 50)
    pdf.cell(0, 15, "SugarCane AI Agent", ln=True, align="C")
    pdf.set_font("Arial", "", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Arquitectura interna: Docker, Docker Hub y despliegue", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
        _w(pdf),
        7,
        "Documento de referencia que explica cómo se relacionan el código fuente, "
        "Docker local, GitHub, Docker Hub, archivos locales (.env, modelo) y el flujo "
        "de alertas por correo.",
    )
    pdf.ln(4)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, "Proyecto: detección de enfermedades foliares en caña de azúcar", ln=True, align="C")

    # 1
    pdf.add_page()
    section_title(pdf, "1. Idea central")
    body(
        pdf,
        "Docker Hub NO reemplaza todo el proyecto. Es solo un almacén remoto donde se "
        "guarda la imagen Docker ya empaquetada de la aplicación.",
    )
    body(
        pdf,
        "Flujo resumido: El código vive en GitHub → se empaqueta en una imagen Docker → "
        "esa imagen se publica en Docker Hub → cualquier PC puede descargarla (docker pull) "
        "y ejecutarla, pero siempre necesita archivos locales: el modelo best.pt, el archivo "
        ".env con secretos, y la carpeta data/ para predicciones y conversaciones.",
    )

    section_title(pdf, "2. Las cuatro piezas del sistema")
    bullet(pdf, "Tu carpeta del proyecto: código fuente (src/), tests, docker/, README.")
    bullet(
        pdf,
        "Docker local (docker compose): construye y ejecuta la app en TU PC usando el "
        "mismo Dockerfile que GitHub.",
    )
    bullet(
        pdf,
        "GitHub: guarda el código y ejecuta workflows automáticos (CI pruebas, CD publica imagen).",
    )
    bullet(
        pdf,
        "Docker Hub: repositorio público de imágenes. Nombre del proyecto: "
        "willianalbertorein/trabajofinalproyecto3:latest",
    )
    body(pdf, "Las cuatro piezas usan el MISMO Dockerfile; no son tecnologías distintas.")

    section_title(pdf, "3. Qué va DENTRO de la imagen Docker")
    bullet(pdf, "Código Python: clasificador, Streamlit, RAG, agente IA, alertas SMTP.")
    bullet(pdf, "Dependencias instaladas: PyTorch, Streamlit, etc.")
    bullet(pdf, "Tests y scripts (entrypoint, informe diario).")
    bullet(pdf, "Carpetas vacías preparadas: /app/models y /app/artifacts.")
    section_title(pdf, "3.1 Qué va FUERA de la imagen (siempre local)")
    bullet(pdf, "models/best.pt — el modelo entrenado (muy pesado, no va a Git ni a la imagen).")
    bullet(pdf, ".env — claves secretas: OPENAI_API_KEY, SMTP_USER, SMTP_APP_PASSWORD.")
    bullet(pdf, "data/predictions/ — registro de cada clasificación (JSONL).")
    bullet(pdf, "data/conversations/ — historial del chat con el agente.")
    body(
        pdf,
        "Por eso, aunque la imagen esté en Docker Hub, cada persona que la use debe "
        "tener el modelo y el .env en su computador.",
    )

    # 2
    pdf.add_page()
    section_title(pdf, "4. Dos formas de ejecutar la misma aplicación")
    section_title(pdf, "4.1 En tu PC (desarrollo) — docker compose")
    body(pdf, "Comandos:")
    code_block(pdf, "  docker compose build app\n  docker compose up app")
    bullet(pdf, "Construye la imagen desde tu carpeta local (código más reciente).")
    bullet(pdf, "Lee el archivo .env de tu disco.")
    bullet(pdf, "Monta models/, artifacts/ y data/ desde tu disco.")
    bullet(pdf, "Abre la app en http://localhost:8501")
    section_title(pdf, "4.2 Desde Docker Hub (otra PC o despliegue)")
    body(pdf, "Comandos:")
    code_block(
        pdf,
        "  docker pull willianalbertorein/trabajofinalproyecto3:latest\n"
        "  docker run -p 8501:8501 -v models:/app/models:ro\n"
        "    -v data/predictions:/app/data/predictions --env-file .env\n"
        "    willianalbertorein/trabajofinalproyecto3:latest streamlit",
    )
    bullet(pdf, "No necesita clonar el repositorio ni compilar.")
    bullet(pdf, "Descarga la imagen ya construida desde internet.")
    bullet(pdf, "Igual necesita models/best.pt y .env en esa PC.")

    section_title(pdf, "5. Por qué se actualiza Docker local primero")
    body(
        pdf,
        "Cuando se cambia el código (por ejemplo, pasar de SendGrid a Gmail SMTP), el orden es:",
    )
    bullet(pdf, "1. Se modifican archivos en src/ (alert_service.py, settings.py, etc.).")
    bullet(pdf, "2. Tú ejecutas: docker compose build app (reconstruye imagen local).")
    bullet(pdf, "3. docker compose up app — pruebas en localhost.")
    bullet(pdf, "4. commit + push a GitHub en la rama main.")
    bullet(pdf, "5. GitHub Actions ejecuta cd-dockerhub.yml y sube imagen nueva a Docker Hub.")
    body(
        pdf,
        "Si no has hecho push después de un cambio, Docker Hub sigue con la versión anterior. "
        "Local y Hub son la misma receta (Dockerfile), pero imágenes construidas en momentos distintos.",
    )

    # 3
    pdf.add_page()
    section_title(pdf, "6. GitHub Actions — workflows automáticos")
    body(pdf, "Cada push a main dispara tres workflows:")
    bullet(pdf, "CI (ci.yml): Ruff lint + pytest + build Docker + tests dentro del contenedor.")
    bullet(
        pdf,
        "CD Publish (cd-publish.yml): publica imagen en GHCR "
        "(ghcr.io/willianreinag/deteccion_enfermedades_foliares_ca-aazucar).",
    )
    bullet(
        pdf,
        "CD Docker Hub (cd-dockerhub.yml): publica en Docker Hub. Requiere secrets "
        "DOCKERHUB_USERNAME y DOCKERHUB_TOKEN en GitHub.",
    )
    body(pdf, "Los tres usan el mismo docker/Dockerfile.")

    section_title(pdf, "7. Flujo interno al clasificar una hoja (con alerta)")
    bullet(pdf, "1. Usuario sube imagen en Streamlit y pulsa Clasificar hoja.")
    bullet(pdf, "2. SugarCanePredictor ejecuta el modelo (best.pt montado en /app/models).")
    bullet(pdf, "3. prediction_logger guarda resultado en data/predictions/predictions.jsonl.")
    bullet(pdf, "4. Si clase != Healthy y confianza >= 0.5 → alert_service envía correo vía Gmail SMTP.")
    bullet(pdf, "5. Variables SMTP leídas del .env inyectado por Docker al arrancar el contenedor.")
    bullet(pdf, "6. Se muestra resultado, Grad-CAM y el chat con el agente IA puede usar la predicción.")

    section_title(pdf, "8. Configuración de correo (Gmail SMTP)")
    bullet(pdf, "SMTP_HOST=smtp.gmail.com  |  SMTP_PORT=587")
    bullet(pdf, "SMTP_USER=tu@gmail.com  |  SMTP_APP_PASSWORD=contraseña de aplicación de Google")
    bullet(pdf, "ALERT_EMAIL=destino del correo  |  ALERT_FROM=normalmente el mismo Gmail")
    bullet(pdf, "Requiere verificación en 2 pasos y contraseña de aplicación en Google Account.")
    bullet(pdf, "Informe diario manual: docker compose run --rm app daily-report")

    section_title(pdf, "9. Tabla resumen — qué está en Docker Hub")
    pdf.set_font("Arial", "B", 10)
    pdf.cell(95, 8, "Componente", border=1)
    pdf.cell(75, 8, "¿En la imagen Docker Hub?", border=1, ln=True)
    pdf.set_font("Arial", "", 10)
    rows = [
        ("Código app (Streamlit, alertas, RAG)", "Sí"),
        ("Dependencias Python", "Sí"),
        ("models/best.pt", "No — se monta localmente"),
        (".env (OpenAI, Gmail SMTP)", "No — nunca subir a Git/Hub"),
        ("data/predictions y conversations", "No — datos en tu PC"),
        ("Envío de correos Gmail", "No — en vivo vía SMTP al clasificar"),
    ]
    for col1, col2 in rows:
        pdf.cell(95, 7, col1, border=1)
        pdf.cell(75, 7, col2, border=1, ln=True)

    # 4
    pdf.add_page()
    section_title(pdf, "10. Analogía simple")
    bullet(pdf, "Código fuente = receta del restaurante (GitHub).")
    bullet(pdf, "Imagen Docker = comida empaquetada lista (Docker Hub).")
    bullet(pdf, ".env + modelo = ingredientes secretos que cada cliente aporta al servir.")
    bullet(pdf, "docker compose en tu PC = cocinar en casa para probar.")
    bullet(pdf, "docker pull desde Hub = pedir el mismo paquete ya preparado.")

    section_title(pdf, "11. Google Cloud (futuro)")
    body(
        pdf,
        "Cuando tengas cuenta GCP, la guía docs/GCP_SETUP.md describe Cloud Run (URL pública), "
        "BigQuery (registro en la nube), Cloud Scheduler (informe 18:00 America/Bogota). "
        "En la nube la función de informe puede usar SendGrid; en local/Docker se usa Gmail SMTP.",
    )

    section_title(pdf, "12. Comandos de referencia rápida")
    cmds = [
        "docker compose build app && docker compose up app",
        "docker compose run --rm test",
        "docker compose run --rm app daily-report",
        "docker compose down",
        "docker pull willianalbertorein/trabajofinalproyecto3:latest",
    ]
    for c in cmds:
        code_block(pdf, f"  {c}")
    pdf.set_font("Arial", "I", 9)
    pdf.ln(4)
    pdf.cell(0, 8, "Documento generado automáticamente — SugarCane Proyecto Final", align="C")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    print(f"PDF creado: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
