import os

for root, dirs, files in os.walk('results/'):
    for f in files:
        if str(f).endswith('.txt'):
            os.remove(f'{root}/{f}')
