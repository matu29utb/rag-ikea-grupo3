import io

import pandas as pd
from langchain_core.documents import Document


class IKEACatalogParser:
    """
    Clase encargada de ingerir, limpiar y reparar el dataset tabular corrupto de IKEA
    para su posterior indexacion en una base de datos vectorial.
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path: str = csv_path

    def _load_and_repair_csv(self) -> pd.DataFrame:
        """
        Lee el archivo de texto linea por linea y repara la estructura del CSV
        antes de pasarselo a Pandas.
        """
        with open(self.csv_path, "r", encoding="utf-8") as f:
            lines: list[str] = f.readlines()

        cleaned_lines: list[str] = []
        for line in lines:
            # Eliminamos saltos de linea y la "basura" final (;;;)
            line = line.strip().rstrip(";")

            # Si la linea entera esta envuelta en comillas (error de exportacion), las quitamos
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]

            # Las descripciones usan comillas dobles (""). Las pasamos a comillas normales (")
            line = line.replace('""', '"')

            cleaned_lines.append(line)

        # Convertimos las lineas limpias a un buffer en memoria
        csv_buffer: io.StringIO = io.StringIO("\n".join(cleaned_lines))

        # Leemos el CSV ya reparado
        df: pd.DataFrame = pd.read_csv(csv_buffer, sep=",", on_bad_lines="skip")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia los datos numericos y maneja valores nulos de forma robusta.
        """
        # Limpiamos posibles espacios en los nombres de las columnas
        df.columns = df.columns.str.strip()

        # Convertimos dimensiones y precio a numerico, rellenando NaN con 0.0
        dimension_cols: list[str] = ["depth", "height", "width", "price"]
        for col in dimension_cols:
            if col in df.columns:
                # Quitamos posibles strings residuales en los precios antes de convertir
                if col == "price":
                    df[col] = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)  # type: ignore[union-attr]

        # Rellenamos el resto de valores nulos con strings vacios
        df = df.fillna("")

        return df

    def parse_to_documents(self) -> list[Document]:
        """
        Convierte cada fila del DataFrame en un Documento de LangChain.
        """
        # Usamos nuestro reparador personalizado
        df: pd.DataFrame = self._load_and_repair_csv()
        df = self._clean_data(df)

        documents: list[Document] = []

        for _, row in df.iterrows():
            # Forzamos string para evitar errores con nulos
            desc: str = str(row.get("short_description", "")).strip()
            nombre: str = str(row.get("name", "Desconocido")).strip()

            # Extraemos en cm y pasamos a metros
            width_m: float = float(row.get("width") or 0.0) / 100.0
            depth_m: float = float(row.get("depth") or 0.0) / 100.0

            # Calculamos el area en m2
            area_m2: float = round(width_m * depth_m, 4)

            # Creamos el contenido semantico para el embedding
            page_content = (
                f"Producto: {nombre}. "
                f"Categoria: {row.get('category', 'General')}. "
                f"Descripcion: {desc}. "
                f"Precio: {float(row.get('price') or 0.0)} euros. "
                f"Medidas (Ancho x Profundo x Alto): {float(row.get('width') or 0.0)} x {float(row.get('depth') or 0.0)} x {float(row.get('height') or 0.0)} cm. "
                f"Area: {area_m2} metros cuadrados. "
                f"Diseñador: {row.get('designer', 'IKEA')}. "
                f"Colores adicionales disponibles: {row.get('other_colors', 'No')}."
            )

            # Extraemos metadatos exactos para filtrado
            metadata: dict[str, str | float] = {
                "item_id": str(row.get("item_id", "")),
                "name": nombre,
                "category": str(row.get("category", "")),
                "price": float(row.get("price") or 0.0),
                "width": float(row.get("width") or 0.0),
                "height": float(row.get("height") or 0.0),
                "depth": float(row.get("depth") or 0.0),
                "area_m2": area_m2,
                "link": str(row.get("link", "")),
                "source": "ikea_database",
            }

            doc: Document = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

        return documents
