import re
import emoji
import string
def clean_text_func(text):
    ## Eliminar URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+', '', text)

    ## Eliminar menciones de usuarios
    text = re.sub(r'@[^\s]+', '', text)

    ## Eliminando los Hashtags
    text = re.sub(r'#\w+', '', text)

    ## Eliminando los emojis
    text = emoji.replace_emoji(text, replace="")

    ## Eliminar dominios .com (palabras terminadas en .com)
    text = re.sub(r'\b\S*\.com\b', '', text)

    # Eliminar caracteres no ASCII (como â\x99«)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    ## Convirtiendo % en palabra percent
    text = re.sub(r'%', ' percent', text)

    ## Eliminando signos de puntuación
    text = text.translate(str.maketrans('', '', string.punctuation + '¡¿'))
    # text = re.sub(rf"[{re.escape(string.punctuation)}]", ' ', text)

    ## Cambiando el doble espacio a espacio simple
    # text = text.replace('  ', ' ')
    text = re.sub(r'\s+', ' ', text)

    ## Eliminando los números del texto
    text = re.sub(r'\d+', '', text)
    # text = re.sub(r'\b\d+([-/]\d+)*\b', '', text)

    ## Eliminando espacios al inicio y final
    text = text.strip()

    ## Convirtiendo a Minúsculas
    text = text.lower()

    return text