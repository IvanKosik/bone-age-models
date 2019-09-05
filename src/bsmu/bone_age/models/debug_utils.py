def print_info(image, prefix: str = ''):
    print(f'{prefix}\t\t| shape: {image.shape}\t| min: {image.min()}\t | max: {image.max()}\t | type: {image.dtype}')


def print_title(title: str):
    print(f'\n\t=== {title.upper()} ===')
