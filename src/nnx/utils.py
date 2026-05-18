from prettytable import PrettyTable

class Utils:
    @staticmethod
    def print_tree(tree, level=0):
        if not isinstance(tree, dict):
            return
        
        max_key_len = max(len(key) for key in tree.keys())
        
        for key, val in tree.items():
            if isinstance(val, dict):
                print(' ' * level * 4 + f'[-] {key}: ')
                
                Utils.print_tree(val, level + 1)
            else:
                print(' ' * level * 4 + f'[+] {key.ljust(max_key_len)} : {val}')
                
    @staticmethod
    def print_table(data: dict, header: bool = True, title: str = None):
        table = PrettyTable(['Key', 'Value'])
        
        table.header = header
        
        if title is not None:
            table.title = title
        
        for key, val in data.items():
            table.add_row([key, val])
            
        print(table)
        
    @staticmethod
    def flatten_dict(data: dict, parent_key='', sep='.'):
        flattened = []
        for key, val in data.items():
            flattened_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(val, dict):
                flattened.extend(Utils.flatten_dict(val, flattened_key, sep=sep).items())
            else:
                flattened.append((flattened_key, val))
                
        return dict(flattened)