import fitz

doc = fitz.open(r'D:\code\PycharmProjects\pytorch\Topo_model\doc\Dismantling Complex Networks by a Neural Model Trained from Tiny Networks.pdf')
print('Total pages:', len(doc))
# Find pages with training loss formula
for i, page in enumerate(doc):
    text = page.get_text()
    if any(kw in text for kw in ['MSE', 'mean square', 'BCE', 'binary cross', 'cross-entropy',
                                  'ListNet', 'ListMLE', 'lambdaloss', 'LambdaRank',
                                  'Huber', 'regression', 'score regression',
                                  'L(', 'L =', 'loss =', 'minimize', 'argmin', 'training objective',
                                  'score label', 'normalized rank', 'ground truth label']):
        print(f'--- page {i+1} ---')
        print(text)
        print()
