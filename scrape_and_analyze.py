

import os
import re
import asyncio
import time
import random
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import tiktoken
from langchain_groq import ChatGroq

# Asumo que tienes estos módulos
from crawler_wrapper import scrap_general 


class ScrapeAndAnalyze:
    """
    Clase modular para hacer scraping de sitios web y análisis de contenido usando LLM.
    """
    
    def __init__(
        self,
        models: List[str] = None,
        token_limit: int = 5000,
        chunk_size_target: int = 4100,
        overlap_paragraphs: int = 2,
        safety_margin: int = 1250,
        wait_between_chunks: int = 20,
        max_wait_for_switch: int = 10 * 60,
        temperature: float = 0
    ):
        """
        Inicializa la clase ScrapAndAnalyze.
        
        Args:
            models: Lista de modelos a usar en orden de preferencia
            token_limit: Límite de tokens para el LLM
            chunk_size_target: Tamaño objetivo de cada chunk
            overlap_paragraphs: Párrafos de solapamiento entre chunks
            safety_margin: Margen de seguridad para tokens
            wait_between_chunks: Tiempo de espera entre chunks exitosos
            max_wait_for_switch: Tiempo máximo antes de cambiar modelo
            temperature: Temperatura del modelo LLM
        """
        self.models = models or [
            'openai/gpt-oss-120b',
            'openai/gpt-oss-20b',
            "deepseek-r1-distill-llama-70b",
            "qwen/qwen3-32b",
            # "moonshotai/kimi-k2-instruct"
        ]
        
        self.current_model_index = 0
        self.token_limit = token_limit
        self.chunk_size_target = chunk_size_target
        self.overlap_paragraphs = overlap_paragraphs
        self.safety_margin = safety_margin
        self.wait_between_chunks = wait_between_chunks
        self.max_wait_for_switch = max_wait_for_switch
        self.temperature = temperature
        
        # Instancia del LLM actual - se inicializa cuando se necesita
        self.current_llm = None
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Prompts por defecto (pueden ser sobrescritos)
        self.example_template = "Aún no se ha prooporcionado un ejemplo"
        self.partial_prompt_template = f"""Tu tarea es analizar un fragmento de texto extraído de un sitio web.
        Extrae y resume la información más relevante. No inventes información.
        Este es solo un trozo del contenido total, por lo que la información puede ser incompleta. 
        Céntrate únicamente en lo que encuentres en el texto proporcionado.
        Si se ha proporcionado un ejemplo, puedes utilizarlo para guiarte en el proceso: {self.example_template}"""
        self.final_prompt_template = f"""Eres un analista experto, y has recibido varios resúmenes parciales extraídos de diferentes páginas de un sitio web.
        Tu objetivo es consolidar toda esta información en un único resumen final, bien estructurado, coherente y sin redundancias.
        Si se ha proporcionado un ejemplo, puedes utilizarlo para guiarte en el proceso: {self.example_template}"""
    
    
    
    def clean_llm_response(self, response: str) -> str:
        """
        Limpia la respuesta del LLM eliminando bloques de pensamiento.
        
        Args:
            response: Respuesta cruda del LLM
            
        Returns:
            Respuesta limpia sin bloques de pensamiento
        """
        if not response:
            return response
        
        # Patrón para detectar bloques <think>...</think>
        # Usa re.DOTALL para que . incluya saltos de línea
        pattern = r'<think>.*?</think>'
        cleaned_response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Limpiar espacios en blanco extra que puedan quedar
        cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
        cleaned_response = cleaned_response.strip()
        
        return cleaned_response
    
    
    
    def set_prompts(self, partial_prompt: str, final_prompt: str, example: str):
        """
        Establece los prompts personalizados para el análisis.
        
        Args:
            partial_prompt: Template para análisis parcial (debe contener {organizacion}, {ejemplo}, {contenido_chunk})
            final_prompt: Template para análisis final (debe contener {organizacion}, {web_oficial}, {ejemplo}, {resumenes_parciales})
            example: Ejemplo de estructura para el análisis
        """
        self.partial_prompt_template = partial_prompt
        self.final_prompt_template = final_prompt
        self.example_template = example
    
    
    
    def _initialize_current_llm(self):
        """
        Inicializa el LLM actual. Solo se llama cuando es necesario crear una nueva instancia.
        """
        model_name = self.models[self.current_model_index]
        print(f"\n--- Inicializando modelo: {model_name} (Modelo {self.current_model_index + 1}/{len(self.models)}) ---")
        self.current_llm = ChatGroq(model=model_name, temperature=self.temperature)
        return self.current_llm
    
    
    
    def switch_to_next_model(self):
        """
        Cambia al siguiente modelo en la lista de forma cíclica y crea una nueva instancia.
        """
        modelo_anterior = self.models[self.current_model_index]
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        modelo_nuevo = self.models[self.current_model_index]
        print(f"*** CAMBIO DE MODELO: De '{modelo_anterior}' a '{modelo_nuevo}' ***")
        
        # Crear nueva instancia del LLM
        self._initialize_current_llm()
        return modelo_nuevo
    
    
    
    def get_current_llm(self):
        """
        Obtiene el LLM actual o crea uno nuevo si no existe.
        """
        if self.current_llm is None:
            return self._initialize_current_llm()
        return self.current_llm
    
    
    
    def invoke_llm_with_retries(self, prompt: str, chunk_info: str, max_retries: int = 10, initial_delay: int = 20):
        """
        Invoca al LLM con lógica de reintentos robusta.
        """
        retries = 0
        delay = initial_delay
        
        while retries < max_retries:
            try:
                llm_instance = self.get_current_llm()
                print(f"    -> Intentando procesar {chunk_info} (intento {retries + 1}/{max_retries})...")
                response = llm_instance.invoke(prompt)
                
                # Limpiar la respuesta antes de devolverla
                cleaned_content = self.clean_llm_response(response.content)
                print(f"    -> {chunk_info} procesado con éxito.")
                return cleaned_content
                
            except Exception as e:
                retries += 1
                print(f"    -> ERROR al procesar {chunk_info}: {e}")

                retry_after = None
                if hasattr(e, 'response') and hasattr(e.response, 'headers') and 'retry-after' in e.response.headers:
                    try:
                        retry_after = int(e.response.headers['retry-after'])
                        print(f"    -> API sugiere esperar {retry_after} segundos.")
                    except (ValueError, TypeError):
                        print("    -> No se pudo convertir 'retry-after' a un número entero.")
                
                if retry_after is not None:
                    if retry_after > self.max_wait_for_switch:
                        print(f"    -> El tiempo de espera ({retry_after}s) supera el umbral de {self.max_wait_for_switch}s.")
                        print("    -> Cambiando de modelo en lugar de esperar.")
                        self.switch_to_next_model()
                        continue
                    else:
                        jitter = random.uniform(0, 5)
                        wait_time = retry_after + jitter
                        print(f"    -> Reintentando en {wait_time:.2f} segundos...")
                        time.sleep(wait_time)
                elif retries < max_retries:
                    # Si es un error persistente y hemos reintentado varias veces, cambiar modelo
                    if retries >= 3:
                        print(f"    -> Error persistente después de {retries} intentos. Cambiando de modelo.")
                        self.switch_to_next_model()
                    else:
                        print(f"    -> Error no relacionado con rate-limit. Esperando {delay} segundos antes de reintentar.")
                        time.sleep(delay)
                        delay *= 2
                else:
                    break
                    
        print(f"    -> Se alcanzó el número máximo de reintentos para {chunk_info}. Saltando este chunk.")
        return None
    
    
    
    def split_text_with_overlap(self, text: str, max_tokens: int, overlap_paragraphs: int) -> List[str]:
        """
        Divide el texto en chunks con solapamiento entre párrafos.
        """
        chunks = []
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return []
        
        current_chunk_paragraphs, current_tokens = [], 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_tokens = len(self.tokenizer.encode(paragraph))
            
            if paragraph_tokens > max_tokens:
                if current_chunk_paragraphs:
                    chunks.append("\n\n".join(current_chunk_paragraphs))
                current_chunk_paragraphs, current_tokens = [], 0
                
                # Dividir párrafo largo por palabras
                words = paragraph.split(' ')
                sub_chunk_words, sub_chunk_tokens = [], 0
                
                for word in words:
                    word_tokens = len(self.tokenizer.encode(' ' + word))
                    if sub_chunk_tokens + word_tokens > max_tokens:
                        chunks.append(' '.join(sub_chunk_words))
                        sub_chunk_words, sub_chunk_tokens = [word], word_tokens
                    else:
                        sub_chunk_words.append(word)
                        sub_chunk_tokens += word_tokens
                
                if sub_chunk_words:
                    chunks.append(' '.join(sub_chunk_words))
                continue
            
            if current_tokens + paragraph_tokens > max_tokens:
                chunks.append("\n\n".join(current_chunk_paragraphs))
                overlap_paragraphs_list = current_chunk_paragraphs[-overlap_paragraphs:]
                current_chunk_paragraphs = overlap_paragraphs_list + [paragraph]
                current_tokens = len(self.tokenizer.encode("\n\n".join(current_chunk_paragraphs)))
            else:
                current_chunk_paragraphs.append(paragraph)
                current_tokens += paragraph_tokens
        
        if current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))
        
        return chunks
    
    
    
    async def analyze_organization(
        self,
        url: str,
        organization_name: str = None,
        max_pages: int = 50,
        max_depth: int = 2,
        save_scraping: bool = True,
        save_scraping_directory: str = "resultados/scraped_docs",
        save_summary: bool = True,
        save_summary_directory: str = "resultados/resumenes",
        return_result: bool = True
    ) -> Optional[str]:
        """
        Analiza una organización mediante scraping y análisis con LLM.
        
        Args:
            url: URL del sitio web a analizar
            organization_name: Nombre de la organización
            max_pages: Número máximo de páginas a scrapear
            max_depth: Profundidad máxima de scraping
            save_scraping: Si guardar el contenido scrapeado
            save_scraping_directory: Directorio donde guardar el scraping
            save_summary: Si guardar el resumen final
            save_summary_directory: Directorio donde guardar el resumen
            return_result: Si retornar el resultado del análisis final
            
        Returns:
            El análisis final como string si return_result=True, None en caso contrario
        """
            
        
        if not all([self.partial_prompt_template, self.final_prompt_template, self.example_template]):
            raise ValueError("Debe establecer los prompts usando set_prompts() antes de analizar")
        
        if not organization_name:
            organization_name = url
            print(f"Procesando URL: {url}")
        else:
            print(f"Procesando organización: {organization_name} - URL: {url}")
        
        # Scraping - pasar save_path solo si save_scraping es True
        if save_scraping:
            os.makedirs(save_scraping_directory, exist_ok=True)
            content_list = await scrap_general(url, max_pages=max_pages, max_depth=max_depth, result_type="cleaned_markdown", save_path=save_scraping_directory)
        else:
            content_list = await scrap_general(url, max_pages=max_pages, max_depth=max_depth, result_type="cleaned_markdown")
        combined_content = "\n\n--- NUEVA PÁGINA ---\n\n".join(content_list)

        if not combined_content.strip():
            print(f"No se pudo obtener contenido para {organization_name}. Saltando...")
            return None if return_result else None
                
        # MAP: Análisis parcial
        partial_summaries = await self._process_chunks(organization_name, combined_content)
        
        if not partial_summaries:
            print("No se pudieron generar resúmenes parciales.")
            return None if return_result else None
        
        # REDUCE: Consolidación final
        final_report = await self._consolidate_summaries(organization_name, url, partial_summaries)
        
        if final_report:
            # Guardar resumen final si está configurado
            if save_summary:
                os.makedirs(save_summary_directory, exist_ok=True)
                summary_file_path = os.path.join(save_summary_directory, f'{organization_name}_resumen.txt')
                print(f"  -> Guardando resumen final en {summary_file_path}")
                with open(summary_file_path, 'w', encoding='utf-8') as f:
                    f.write(final_report)
        
        # Retornar resultado solo si return_result es True
        return final_report if return_result else None
    
    
    
    async def _process_chunks(self, organization_name: str, content: str) -> List[str]:
        """
        Procesa el contenido en chunks y genera resúmenes parciales.
        """
        # Calcular tokens disponibles para contenido
        partial_prompt_base = self.partial_prompt_template.format(
            organizacion=organization_name, 
            ejemplo=self.example_template, 
            contenido_chunk=""
        )
        prompt_tokens = len(self.tokenizer.encode(partial_prompt_base))
        max_content_tokens = self.token_limit - prompt_tokens - self.safety_margin
        
        # Dividir en chunks
        chunks = self.split_text_with_overlap(content, max_content_tokens, self.overlap_paragraphs)
        
        if chunks:
            print(f"  -> Contenido dividido en {len(chunks)} chunks. Procesando...")
        
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            prompt = self.partial_prompt_template.format(
                organizacion=organization_name,
                ejemplo=self.example_template,
                contenido_chunk=chunk
            )
            
            summary_content = self.invoke_llm_with_retries(
                prompt, f"chunk de contenido {i + 1}/{len(chunks)}"
            )
            
            if summary_content:
                partial_summaries.append(summary_content)
                print(f"    -> Esperando {self.wait_between_chunks} segundos antes del siguiente chunk...")
                time.sleep(self.wait_between_chunks)
        
        return partial_summaries
    
    
    
    async def _consolidate_summaries(self, organization_name: str, url: str, summaries: List[str]) -> Optional[str]:
        """
        Consolida los resúmenes parciales en un informe final.
        """
        print("  -> Unificando resúmenes parciales en un informe final...")
        current_summaries = summaries
        
        while True:
            combined_summaries = "\n\n--- NUEVO RESUMEN PARCIAL ---\n\n".join(current_summaries)
            
            # Verificar si cabe en el límite de tokens
            final_prompt_base = self.final_prompt_template.format(
                organizacion=organization_name,
                web_oficial=url,
                ejemplo=self.example_template,
                resumenes_parciales=""
            )
            prompt_tokens = len(self.tokenizer.encode(final_prompt_base))
            summaries_tokens = len(self.tokenizer.encode(combined_summaries))
            total_tokens = prompt_tokens + summaries_tokens
            
            if total_tokens < self.token_limit - self.safety_margin:
                print(f"  -> Consolidación final lista ({total_tokens} tokens). Generando informe.")
                break
            
            # Aplicar otra capa de reducción
            print(f"  -> La consolidación es muy larga ({total_tokens} tokens). Aplicando nueva capa de reducción.")
            max_content_tokens = self.token_limit - prompt_tokens - self.safety_margin
            summary_chunks = self.split_text_with_overlap(combined_summaries, max_content_tokens, self.overlap_paragraphs)
            
            if summary_chunks:
                print(f"  -> Resúmenes divididos en {len(summary_chunks)} chunks para reducir.")
            
            next_level_summaries = []
            for i, summary_chunk in enumerate(summary_chunks):
                prompt = self.final_prompt_template.format(
                    organizacion=organization_name,
                    web_oficial=url,
                    ejemplo=self.example_template,
                    resumenes_parciales=summary_chunk
                )
                
                intermediate_summary = self.invoke_llm_with_retries(
                    prompt, f"chunk de resumen {i + 1}/{len(summary_chunks)}"
                )
                
                if intermediate_summary:
                    next_level_summaries.append(intermediate_summary)
                    print(f"    -> Esperando {self.wait_between_chunks} segundos...")
                    time.sleep(self.wait_between_chunks)
            
            if not next_level_summaries:
                print("  -> Error: La capa de reducción no produjo resúmenes. Abortando.")
                return None
            
            current_summaries = next_level_summaries
        
        # Generar informe final
        final_prompt = self.final_prompt_template.format(
            organizacion=organization_name,
            web_oficial=url,
            ejemplo=self.example_template,
            resumenes_parciales=combined_summaries
        )
        
        final_report = self.invoke_llm_with_retries(final_prompt, "informe final consolidado")
        return final_report
    
    


# Ejemplo de uso
if __name__ == '__main__':
    # Definir los prompts (ejemplo basado en tu código original)
    ejemplo = """
Nombre: Amigos de la Tierra Argentina
Aspectos institucionales
    • Estatus legal: Organización civil sin fines de lucro. Formal
    • Año de creación: 1981
    • Misión: ...
    • Visión: ...
    • Destinatarios de su producción y actividades: ...
    • Financiamiento: ...
    • Otros aspectos institucionales: ...
Informe financiero
    • Financiamiento: ...
    • Personal: ...
Producción
    • Sector político en el que se inscribe: ...
    • Temáticas: ...
    • Productos: ...
    • Formatos de productos: ...
Comunicación
    • Espacios donde se comparte la información: ...
    • Página web: ...
    • Publicaciones: ...
    • Participación en redes temáticas: ...
Estructura institucional: ONG
"""

    prompt_parcial = """
Tu tarea es analizar un fragmento de texto extraído del sitio web de la organización "{organizacion}".
Extrae y resume la información más relevante basándote en la estructura del siguiente ejemplo.
No inventes información. Si algo no aparece en el texto, no lo incluyas.
Este es solo un trozo del contenido total, por lo que la información puede ser incompleta. Céntrate únicamente en lo que encuentres en el texto proporcionado.

EJEMPLO DE ESTRUCTURA:
{ejemplo}

---
CONTENIDO DEL FRAGMENTO A ANALIZAR:
{contenido_chunk}
---

Si lo encuentras, en una sección de aspectos institucionales incluye datos sobre los convenios o alianzas estratégicas que tenga la organización
Puedes también comentar datos acerca del personal (rentado, voluntario, etc.)
También, si lo encuentras, comenta si la organización tiene personería jurídica o no y si la misma es formal o no

Extrae la información relevante de este fragmento:
"""

    prompt_final = """
Tu tarea es actuar como un analista experto. Has recibido varios resúmenes parciales extraídos de diferentes páginas del sitio web de la organización "{organizacion}" ({web_oficial}).
Tu objetivo es consolidar toda esta información en un único resumen final, bien estructurado, coherente y sin redundancias.
Usa el siguiente formato como guía.

EJEMPLO DE ESTRUCTURA FINAL:
{ejemplo}

---
RESÚMENES PARCIALES A CONSOLIDAR:
{resumenes_parciales}
---

HAZLO EN TEXTO SIMPLE, NO MARKDOWN
Basándote en los resúmenes anteriores, crea el informe final y consolidado sobre la organización:
"""

    # Crear instancia y configurar
    analyzer = ScrapeAndAnalyze()
    analyzer.set_prompts(prompt_parcial, prompt_final, ejemplo)
    
    # Ejemplo de uso
    async def ejemplo_uso():
        # Ejemplo 1: Solo scraping y análisis, devolver resultado
        # resultado = await analyzer.analyze_organization(
        #     organization_name="Ejemplo CELS",
        #     url="https://www.cels.org.ar/web/",
        #     max_pages=20,
        #     max_depth=2,
        #     return_result=True
        # )
        
        # Ejemplo 2: Guardar scraping y resumen, no devolver resultado  
        await analyzer.analyze_organization(
            organization_name="Ejemplo CELS",
            url="https://www.cels.org.ar/web/",
            max_pages=5,
            max_depth=2,
            save_scraping=True,
            save_scraping_directory="mis_datos/scraping",
            save_summary=True, 
            save_summary_directory="mis_datos/resumenes",
            return_result=False
        )
        
        # print("Análisis completado:", resultado is not None)
    
    asyncio.run(ejemplo_uso())