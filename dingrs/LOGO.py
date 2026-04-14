from PIL import Image
import io
import base64

# 1. Citirea imaginii de la Telegram
binary_data = items[0].binary.data.to_bytes()
base_image = Image.open(io.BytesIO(binary_data)).convert("RGBA")

# 2. Încărcarea Logo-ului tău pătrat
logo_path = "/Users/mario/Desktop/Aladin/LOGO.png" 
logo = Image.open(logo_path).convert("RGBA")

# 3. Redimensionare Inteligentă (High Quality)
img_w, img_h = base_image.size
# Logo-ul va ocupa 18% din lățimea pozei pentru a fi vizibil dar discret
logo_size = int(img_w * 0.18) 
logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)

# 4. Poziționare în colțul dreapta-jos (cu spațiere/padding)
padding = 40
pos = (img_w - logo_size - padding, img_h - logo_size - padding)

# 5. Lipire logo (folosim logo-ul și ca mască pentru a păstra transparența dacă există)
base_image.paste(logo, pos, logo)

# 6. Salvare la calitate maximă pentru a evita pixelarea
output = io.BytesIO()
# Convertim în RGB pentru formatul final JPEG, păstrând culorile vii
base_image.convert("RGB").save(output, format="JPEG", quality=98, subsampling=0)

return [{
    "binary": {
        "data": base64.b64encode(output.getvalue()).decode('utf-8'),
        "fileName": "branded_final.jpg",
        "mimeType": "image/jpeg"
    }
}]