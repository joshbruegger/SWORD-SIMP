import gdown

to_download = [
    ["Giovanni_di_Nicola_Madonna_in_trono_col_Bambino_e_donatore.png",
        "https://drive.google.com/file/d/10t4zLgSG4YY1XMWREjmmX4Vlr64AeJMA/view?usp=share_link"],
]

for output, url in to_download:
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)
