
from src.core.helpers import unicode_to_ascii

def unicode_to_ascii_test(self):
    output = unicode_to_ascii('Ślusàrski')
    self.assertEqual(output, 'Slusarski')
    
#    category_lines, all_categories = load_data()
#    print(category_lines['Italian'][:5])
    
#    print(letter_to_tensor('J')) # [1, 57]
#    print(line_to_tensor('Jones').size()) # [5, 1, 57]