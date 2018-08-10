import configparser as cp

def parser_cfg_file(cfg_file):
    net_params = {}
    train_params = {}

    config = cp.ConfigParser()
    config.read(cfg_file)

    for section in config.sections():
        # 获取配置文件中的net信息
        if section == 'net':
            for option in config.options(section):
                net_params[option] = config.get(section, option)

        # 获取配置文件中的train信息
        if section == 'train':
            for option in config.options(section):
                train_params[option] = config.get(section, option)

    return net_params,train_params

if __name__=='__main__':
    net_params, train_params = parser_cfg_file('../net.cfg')

    print(net_params)




