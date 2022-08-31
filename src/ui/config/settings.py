import configparser

class Settings:
    _settings_path = "src\\ui\\config\\settings.ini"
    _config_parser = configparser.ConfigParser()

    _initialized = False

    def _write_config_file(self):
        with open(Settings._settings_path, 'w') as configfile:
            Settings._config_parser.write(configfile)
    
    def _init():
        Settings._config_parser.read(Settings._settings_path)
        Settings._initialized = True

    def get(section, setting):
        if(not Settings._initialized):
            Settings._init()
        return Settings._config_parser.get(section, setting)

    def set(section, setting, value):
        if(not Settings._initialized):
            Settings._init()

        require_file_save = False

        # Populate section if it doesn't exist
        if not Settings._config_parser.has_section(section):
            Settings._config_parser.add_section(section)
            require_file_save = True

        Settings._config_parser.set(section, setting, value)

        if require_file_save:
            Settings._write_config_file()
