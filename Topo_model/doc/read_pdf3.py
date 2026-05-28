import fitz

doc = fitz.open(r'D:\code\PycharmProjects\pytorch\Topo_model\doc\Dismantling Complex Networks by a Neural Model Trained from Tiny Networks.pdf')
# Print full text of pages 4-5 (section 4: MODEL TRAINING)
for i in [3, 4, 5]:  # 0-indexed
    page = doc[i]
    text = page.get_text()
    print(f'=== PAGE {i+1} (full) ===')
    print(text)
    print()
