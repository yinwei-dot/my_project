import fitz

doc = fitz.open(r'D:\code\PycharmProjects\pytorch\Topo_model\doc\Dismantling Complex Networks by a Neural Model Trained from Tiny Networks.pdf')
print('Total pages:', len(doc))
for i, page in enumerate(doc):
    text = page.get_text()
    if any(kw in text for kw in ['loss', 'Loss', 'objective', 'Objective', 'label', 'Label', 'ground truth', 'rank', 'Rank', 'MSE', 'BCE']):
        print(f'--- page {i+1} ---')
        print(text[:3000])
        print()
