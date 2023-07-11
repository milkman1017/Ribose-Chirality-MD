import configparser

sheet_config = configparser.ConfigParser()
sheet_config['Sheet Setup'] = {
    'sheet_length': 5,
    'sheet_width': 5,
    'lconc':5
}
sheet_config['Simulation Setup'] = {
    'number proccesss':1,
    'number gpus': 1,
    'number sims':1,
    'number steps':100000
}
sheet_config['Output Parameters'] = {
    'output directory': ".",
    'report interval':500,
    'verbose':True
}
with open('sheet_config.ini', 'w') as configfile:
    sheet_config.write(configfile)


analysis_config = configparser.ConfigParser()
analysis_config['Input Setup'] = {

}
analysis_config['Analyses'] = {

}
analysis_config['Output Parameters'] = {

}
with open('analysis_config.ini', 'w') as configfile:
    sheet_config.write(configfile)


umbrella_config = configparser.ConfigParser()
umbrella_config['Umbrella Setup'] = {

}
umbrella_config['Simulation Parameters'] = {

}
umbrella_config['Output Parameters'] = {

}
with open('umbrella_config.ini', 'w') as configfile:
    sheet_config.write(configfile)