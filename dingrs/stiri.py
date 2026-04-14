import pandas as pd

# Încercăm o sursă de backup (CSV public de pe GitHub care e actualizat des)
url = "https://raw.githubusercontent.com/mdeverna/economic_calendar/master/data/calendar_raw.csv"

try:
    df = pd.read_csv(url)
    print(f"✅ Am găsit {len(df)} știri!")
    print(df.head()) # Vedem primele rânduri
except:
    print("❌ Nici asta nu merge. Să trecem la planul manual.")