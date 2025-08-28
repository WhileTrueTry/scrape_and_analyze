"""
Web Scraper con crawl4ai

Este módulo proporciona funciones para extraer contenido de páginas web de forma
síncrona y asíncrona, con capacidades de scraping simple y recursivo.

Funcionalidades:
- Extracción de página individual (landpage)
- Extracción recursiva de múltiples páginas (general)
- Guardado opcional de contenido en archivos
- Múltiples tipos de resultado (markdown, cleaned_html, fit_html, html, cleaned_markdown)
- Interfaz de línea de comandos

Dependencias:
- crawl4ai
- beautifulsoup4
- asyncio (incluido en Python estándar)

Autor: WhileTrueTry
Versión: 1.2
"""

import os
import re
import sys
import asyncio
import argparse
from urllib.parse import urljoin, urlparse
from typing import List, Optional, Union
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup


def clean_markdown(text: str) -> str:
    """
    Limpia el texto markdown eliminando enlaces en formato [texto](url).
    
    Args:
        markdown_text (str): Texto en formato markdown
        
    Returns:
        str: Texto markdown sin enlaces
        
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Eliminar completamente los enlaces de imágenes: ![alt text](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # 2. Convertir enlaces de texto a solo texto: [texto](url) -> texto
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # 3. Eliminar URLs sueltas (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Eliminar líneas que sean solo separadores o caracteres no alfanuméricos
    # (ej. ---, ***, ====)
    text = re.sub(r'^\s*[-*=_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # 5. Comprimir espacios en blanco y saltos de línea múltiples
    text = re.sub(r'\n\s*\n', '\n\n', text) # Reemplaza múltiples saltos de línea por dos
    text = text.strip() # Elimina espacios al principio y al final
    
    return text


def extract_content_by_type(result, result_type: str) -> str:
    """
    Extrae el contenido según el tipo especificado.
    
    Args:
        result: Resultado del crawler
        result_type (str): Tipo de contenido a extraer
        
    Returns:
        str: Contenido extraído según el tipo especificado
        
    Raises:
        ValueError: Si result_type no es válido
    """
    valid_types = ["markdown", "cleaned_html", "fit_html", "html", "cleaned_markdown"]
    
    if result_type not in valid_types:
        raise ValueError(f"result_type debe ser uno de: {valid_types}")
    
    content = ""
    
    if result_type == "markdown":
        content = result.markdown if hasattr(result, 'markdown') and result.markdown else ""
    elif result_type == "cleaned_html":
        content = result.cleaned_html if hasattr(result, 'cleaned_html') and result.cleaned_html else ""
    elif result_type == "fit_html":
        content = result.fit_html if hasattr(result, 'fit_html') and result.fit_html else ""
    elif result_type == "html":
        content = result.html if hasattr(result, 'html') and result.html else ""
    elif result_type == "cleaned_markdown":
        # Primero obtener markdown, luego limpiarlo
        raw_markdown = result.markdown if hasattr(result, 'markdown') and result.markdown else ""
        content = clean_markdown(raw_markdown)
    
    return content


async def scrap_landpage(url: str, save_path: Optional[str] = None, return_content: bool = True, 
                        result_type: str = "markdown") -> Optional[str]:
    """
    Extrae el contenido de una sola página principal.
    
    Args:
        url (str): La URL de la página a extraer
        save_path (str, opcional): Nombre del directorio donde guardar el contenido.
                                 Si es None, no se guarda en archivos.
        return_content (bool): Si True, retorna el contenido extraído. Si False, retorna None.
        result_type (str): Tipo de contenido a extraer. Opciones: 
                          "markdown", "cleaned_html", "fit_html", "html", "cleaned_markdown"
        
    Returns:
        Optional[str]: El contenido de la página como string si return_content=True,
                      None si return_content=False.
                      
    Raises:
        Exception: Si hay errores durante la extracción web
        ValueError: Si result_type no es válido
        
    Example:
        >>> content = await scrap_landpage("https://example.com")
        >>> content = await scrap_landpage("https://example.com", result_type="cleaned_html")
        >>> await scrap_landpage("https://example.com", save_path="mi_sitio", return_content=False)
    """
    print(f"[*] Extrayendo página principal: {url}")
    print(f"[*] Tipo de resultado: {result_type}")
    
    # Configurar directorio de guardado si se especifica
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"[*] El contenido se guardará en: {save_path}")
    
    content = ""
    
    try:
        # Usar configuración simple sin parámetros complejos
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(url=url)
            
            # Procesar resultado directamente
            if result.success:
                content = extract_content_by_type(result, result_type)
                
                print(f"[*] Página principal extraída exitosamente. Longitud del contenido: {len(content)} caracteres")
                
                # Guardar en archivo si se especifica la ruta
                if save_path:
                    save_content(content, url, save_path, 1, result_type)
                
            else:
                error_msg = result.error_message if hasattr(result, 'error_message') else "Error desconocido"
                print(f"[!] Error al extraer página principal: {error_msg}")
                return None
                    
    except Exception as e:
        print(f"[!] Error extrayendo página principal: {str(e)}")
        raise
    
    # Retornar contenido solo si se solicita explícitamente
    return content if return_content else None


def extract_links(html_content: str, base_url: str) -> List[str]:
    """
    Extrae todos los enlaces del contenido HTML que pertenecen al mismo dominio.
    
    Args:
        html_content (str): Contenido HTML de la página
        base_url (str): URL base para resolver enlaces relativos
        
    Returns:
        List[str]: Lista de URLs absolutas del mismo dominio, sin duplicados
        
    Note:
        - Convierte URLs relativas a absolutas
        - Filtra solo enlaces del mismo dominio
        - Excluye archivos multimedia y scripts
        - Elimina fragmentos (#) de las URLs
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        # Obtener dominio de la URL para filtrado de dominio
        start_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convertir URLs relativas a absolutas
            absolute_url = urljoin(base_url, href)
            
            # Parsear la URL
            parsed = urlparse(absolute_url)
            
            # Solo incluir enlaces del mismo dominio
            if parsed.netloc == start_domain:
                # Limpiar la URL (remover fragmentos)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    clean_url += f"?{parsed.query}"
                
                # Evitar enlaces comunes que no son contenido
                excluded_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', 
                                     '.css', '.js', '.ico', '.svg', '.woff', '.ttf']
                excluded_schemes = ['mailto:', 'tel:', 'javascript:']
                
                if not any(skip in clean_url.lower() for skip in excluded_extensions + excluded_schemes + ['#']):
                    links.append(clean_url)
        
        return list(set(links))  # Remover duplicados
        
    except Exception as e:
        print(f"[!] Error extrayendo enlaces: {str(e)}")
        return []


def save_content(content: str, url: str, save_dir: str, index: int, result_type: str = "markdown") -> bool:
    """
    Guarda el contenido extraído en un archivo de texto.
    
    Args:
        content (str): Contenido a guardar
        url (str): URL de origen del contenido
        save_dir (str): Directorio donde guardar el archivo
        index (int): Número de índice para el archivo
        result_type (str): Tipo de contenido para incluir en el nombre del archivo
        
    Returns:
        bool: True si se guardó exitosamente, False en caso contrario
        
    Note:
        El nombre del archivo se genera de forma segura desde la URL,
        reemplazando caracteres no válidos por guiones bajos e incluyendo el tipo de resultado.
    """
    try:
        # Crear un nombre de archivo seguro desde la URL
        parsed = urlparse(url)
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', parsed.path.strip('/'))
        if not safe_filename:
            safe_filename = "index"
        
        filename = f"{index:03d}_{safe_filename}_{result_type}.txt"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n")
            f.write(f"Tipo de resultado: {result_type}\n")
            f.write("=" * 80 + "\n\n")
            f.write(content)
        
        print(f"[*] Contenido guardado en: {filename}")
        return True
        
    except Exception as e:
        print(f"[!] Error guardando contenido: {str(e)}")
        return False


async def scrap_general(url: str, max_pages: int = 100, max_depth: int = 3, 
                       save_path: Optional[str] = None, return_content: bool = True,
                       result_type: str = "markdown") -> Optional[List[str]]:
    """
    Extrae múltiples páginas de forma recursiva comenzando desde una URL base.
    
    Args:
        url (str): La URL base desde donde comenzar la extracción
        max_pages (int): Número máximo de páginas a extraer (predeterminado: 100)
        max_depth (int): Profundidad máxima para el rastreo recursivo (predeterminado: 3)
        save_path (str, opcional): Ruta para guardar el contenido extraído. 
                                  Si es None, el contenido no se guarda en archivos.
        return_content (bool): Si True, retorna el contenido extraído. Si False, solo guarda archivos.
        result_type (str): Tipo de contenido a extraer. Opciones: 
                          "markdown", "cleaned_html", "fit_html", "html", "cleaned_markdown"
        
    Returns:
        Optional[List[str]]: Lista del contenido extraído de todas las páginas si return_content=True,
                           None si return_content=False
                           
    Raises:
        Exception: Si hay errores críticos durante la extracción
        ValueError: Si result_type no es válido
        
    Note:
        - Respeta el mismo dominio (no hace scraping de sitios externos)
        - Implementa una pausa entre solicitudes para ser respetuoso
        - Evita duplicados manteniendo registro de URLs visitadas
        - La profundidad 0 es la página inicial, 1 son los enlaces de esa página, etc.
    """
    print(f"[*] Iniciando extracción general desde: {url}")
    print(f"[*] Páginas máximas: {max_pages}, Profundidad máxima: {max_depth}")
    print(f"[*] Tipo de resultado: {result_type}")
    
    # Configurar directorio de guardado si se especifica
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"[*] El contenido se guardará en: {save_path}")
    
    # Inicializar variables de seguimiento
    scraped_content = [] if return_content else None
    visited_urls = set()
    urls_to_scrape = [(url, 0)]  # (url, profundidad)
    crawled_count = 0
    
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            while urls_to_scrape and crawled_count < max_pages:
                current_url, current_depth = urls_to_scrape.pop(0)
                
                # Omitir si ya fue visitada o se excedió la profundidad máxima
                if current_url in visited_urls or current_depth > max_depth:
                    continue
                
                print(f"[*] Extrayendo ({current_depth}/{max_depth}): {current_url}")
                visited_urls.add(current_url)
                
                try:
                    # Extraer la URL actual
                    result = await crawler.arun(url=current_url)
                    
                    if result.success:
                        # Extraer contenido según el tipo especificado
                        content = extract_content_by_type(result, result_type)
                        
                        if return_content and scraped_content is not None:
                            scraped_content.append(content)
                        
                        crawled_count += 1
                        
                        # Guardar en archivo si se especifica la ruta
                        if save_path:
                            save_content(content, current_url, save_path, crawled_count, result_type)
                        
                        print(f"[*] Página extraída exitosamente {crawled_count}: {current_url}")
                        
                        # Extraer enlaces para el siguiente nivel de profundidad (si no está en profundidad máxima)
                        if current_depth < max_depth:
                            # Para extraer enlaces siempre usamos HTML
                            html_for_links = result.html if hasattr(result, 'html') else ""
                            if html_for_links:
                                new_links = extract_links(html_for_links, current_url)
                                
                                # Agregar nuevos enlaces a la cola
                                for link in new_links:
                                    if link not in visited_urls:
                                        urls_to_scrape.append((link, current_depth + 1))
                                
                                print(f"[*] Se encontraron {len(new_links)} nuevos enlaces en profundidad {current_depth + 1}")
                    
                    else:
                        error_msg = result.error_message if hasattr(result, 'error_message') else "Error desconocido"
                        print(f"[!] Error al extraer: {current_url} - {error_msg}")
                
                except Exception as page_error:
                    print(f"[!] Error extrayendo {current_url}: {str(page_error)}")
                    continue
                
                # Pequeña pausa para ser respetuoso con el servidor
                await asyncio.sleep(0.5)
                    
    except Exception as e:
        print(f"[!] Error durante la extracción general: {str(e)}")
        if return_content:
            return scraped_content
        return None
    
    print(f"\n[*] Extracción general finalizada. Se extrajeron exitosamente {crawled_count} páginas.")
    print(f"[*] URLs visitadas: {len(visited_urls)}")
    
    return scraped_content if return_content else None


# Wrappers síncronos de conveniencia
def scrap_landpage_sync(url: str, save_path: Optional[str] = None, return_content: bool = True,
                       result_type: str = "markdown") -> Optional[str]:
    """
    Wrapper síncrono para scrap_landpage.
    
    Args:
        url (str): La URL de la página a extraer
        save_path (str, opcional): Nombre del directorio donde guardar el contenido
        return_content (bool): Si retornar el contenido extraído
        result_type (str): Tipo de contenido a extraer
        
    Returns:
        Optional[str]: El contenido de la página o None si return_content=False
    """
    return asyncio.run(scrap_landpage(url, save_path, return_content, result_type))


def scrap_general_sync(url: str, max_pages: int = 100, max_depth: int = 3, 
                      save_path: Optional[str] = None, return_content: bool = True,
                      result_type: str = "markdown") -> Optional[List[str]]:
    """
    Wrapper síncrono para scrap_general.
    
    Args:
        url (str): La URL base desde donde comenzar la extracción
        max_pages (int): Número máximo de páginas a extraer
        max_depth (int): Profundidad máxima para el rastreo recursivo
        save_path (str, opcional): Ruta para guardar el contenido extraído
        return_content (bool): Si retornar el contenido extraído
        result_type (str): Tipo de contenido a extraer
        
    Returns:
        Optional[List[str]]: Lista del contenido extraído o None
    """
    return asyncio.run(scrap_general(url, max_pages, max_depth, save_path, return_content, result_type))


def main():
    """
    Función principal para la interfaz de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Web Scraper con crawl4ai - Extrae contenido de páginas web",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  Extraer una sola página:
    python scraper.py landpage https://example.com
    
  Extraer una sola página en formato HTML limpio:
    python scraper.py landpage https://example.com --result-type cleaned_html
    
  Extraer una sola página y guardar:
    python scraper.py landpage https://example.com --save-path mi_sitio
    
  Extraer múltiples páginas (recursivo):
    python scraper.py general https://example.com --max-pages 50 --max-depth 2
    
  Extraer en formato markdown limpio y guardar:
    python scraper.py general https://example.com --save-path docs --result-type cleaned_markdown
    
  Solo guardar sin retornar contenido:
    python scraper.py general https://example.com --save-path docs --no-return

Tipos de resultado disponibles:
  - markdown: Contenido en formato markdown (predeterminado)
  - cleaned_html: HTML limpio sin scripts ni estilos
  - fit_html: HTML optimizado para lectura
  - html: HTML completo de la página
  - cleaned_markdown: Markdown sin enlaces [texto](url)
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Modo de extracción')
    
    # Parser para modo landpage
    landpage_parser = subparsers.add_parser('landpage', help='Extraer una sola página')
    landpage_parser.add_argument('url', help='URL de la página a extraer')
    landpage_parser.add_argument('--save-path', help='Directorio donde guardar el contenido')
    landpage_parser.add_argument('--no-return', action='store_true',
                               help='Solo procesar, no retornar contenido')
    landpage_parser.add_argument('--result-type', 
                               choices=['markdown', 'cleaned_html', 'fit_html', 'html', 'cleaned_markdown'],
                               default='markdown',
                               help='Tipo de contenido a extraer (default: markdown)')
    
    # Parser para modo general
    general_parser = subparsers.add_parser('general', help='Extraer múltiples páginas recursivamente')
    general_parser.add_argument('url', help='URL base desde donde comenzar')
    general_parser.add_argument('--max-pages', type=int, default=100, 
                               help='Número máximo de páginas a extraer (default: 100)')
    general_parser.add_argument('--max-depth', type=int, default=3,
                               help='Profundidad máxima de rastreo (default: 3)')
    general_parser.add_argument('--save-path', help='Directorio donde guardar el contenido')
    general_parser.add_argument('--no-return', action='store_true',
                               help='Solo guardar archivos, no retornar contenido')
    general_parser.add_argument('--result-type', 
                               choices=['markdown', 'cleaned_html', 'fit_html', 'html', 'cleaned_markdown'],
                               default='markdown',
                               help='Tipo de contenido a extraer (default: markdown)')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    try:
        if args.mode == 'landpage':
            print(f"[*] Modo: Extracción de página individual")
            return_content = not args.no_return
            content = scrap_landpage_sync(args.url, args.save_path, return_content, args.result_type)
            
            if content and return_content and not args.save_path:
                print(f"\n[*] Contenido extraído ({len(content)} caracteres):")
                print("-" * 50)
                # Mostrar solo los primeros 500 caracteres como preview
                preview = content[:500] + "..." if len(content) > 500 else content
                print(preview)
                
        elif args.mode == 'general':
            print(f"[*] Modo: Extracción recursiva")
            return_content = not args.no_return
            
            content_list = scrap_general_sync(
                args.url, 
                args.max_pages, 
                args.max_depth, 
                args.save_path,
                return_content,
                args.result_type
            )
            
            if content_list and return_content:
                print(f"\n[*] Se extrajeron {len(content_list)} páginas.")
                total_chars = sum(len(content) for content in content_list)
                print(f"[*] Total de caracteres extraídos: {total_chars}")
                
                if not args.save_path:
                    print("\n[*] Preview del primer contenido:")
                    print("-" * 50)
                    preview = content_list[0][:500] + "..." if len(content_list[0]) > 500 else content_list[0]
                    print(preview)
        
        print("\n[*] Proceso completado exitosamente.")
        
    except KeyboardInterrupt:
        print("\n[!] Proceso interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Error durante la ejecución: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()