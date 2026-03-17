import pandas as pd
import io
from langchain_core.documents import Document
from typing import List

class IKEACatalogParser:
    """
    Clase encargada de ingerir, limpiar y reparar el dataset tabular corrupto de IKEA
    para su posterior indexacion en una base de datos vectorial.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def _load_and_repair_csv(self) -> pd.DataFrame:
        """
        Lee el archivo de texto linea por linea y repara la estructura del CSV
        antes de pasarselo a Pandas.
        """
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cleaned_lines = []
        for line in lines:
            # Eliminamos saltos de linea y la "basura" final (;;;)
            line = line.strip().rstrip(';')

            # Si la linea entera esta envuelta en comillas (error de exportacion), las quitamos
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]

            # Las descripciones usan comillas dobles (""). Las pasamos a comillas normales (")
            line = line.replace('""', '"')

            cleaned_lines.append(line)

        # Convertimos las lineas limpias a un buffer en memoria
        csv_buffer = io.StringIO('\n'.join(cleaned_lines))

        # Leemos el CSV ya reparado
        df = pd.read_csv(csv_buffer, sep=',', on_bad_lines='skip')
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia los datos numericos y maneja valores nulos de forma robusta.
        """
        # Limpiamos posibles espacios en los nombres de las columnas
        df.columns = df.columns.str.strip()

        # Convertimos dimensiones y precio a numerico, rellenando NaN con 0.0
        dimension_cols = ['depth', 'height', 'width', 'price']
        for col in dimension_cols:
            if col in df.columns:
                # Quitamos posibles strings residuales en los precios antes de convertir
                if col == 'price':
                    df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Rellenamos el resto de valores nulos con strings vacios
        df = df.fillna('')

        return df

    def parse_to_documents(self) -> List[Document]:
        """
        Convierte cada fila del DataFrame en un Documento de LangChain.
        """
        # Usamos nuestro reparador personalizado
        df = self._load_and_repair_csv()
        df = self._clean_data(df)

        documents = []

        for _, row in df.iterrows():
            # Forzamos string para evitar errores con nulos
            desc = str(row.get('short_description', '')).strip()
            nombre = str(row.get('name', 'Desconocido')).strip()

            # Extraemos en cm y pasamos a metros
            width_m = float(row.get('width', 0.0)) / 100.0
            depth_m = float(row.get('depth', 0.0)) / 100.0

            # Calculamos el area en m2
            area_m2 = round(width_m * depth_m, 4)

            # Creamos el contenido semantico para el embedding
            page_content = (
                f"Producto: {nombre}. "
                f"Categoria: {row.get('category', 'General')}. "
                f"Descripcion: {desc}. "
                f"Diseñador: {row.get('designer', 'IKEA')}. "
                f"Colores adicionales disponibles: {row.get('other_colors', 'No')}."
            )

            # Extraemos metadatos exactos para filtrado
            metadata = {
                "item_id": str(row.get('item_id', '')),
                "name": nombre,
                "category": str(row.get('category', '')),
                "price": float(row.get('price', 0.0)),
                "width": float(row.get('width', 0.0)),
                "height": float(row.get('height', 0.0)),
                "depth": float(row.get('depth', 0.0)),
                "area_m2": area_m2,
                "link": str(row.get('link', '')),
                "source": "ikea_database"
            }

            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

        return documents