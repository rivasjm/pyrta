import configparser

CONFIG_FILENAME = 'config.ini'


def get_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILENAME)
    return config['mast']


if __name__ == '__main__':
    c = get_config()
    print(c['mast_path'])