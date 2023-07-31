
class NordColor:
    def __init__(self):
        """
        nord0:  #2e3440; dark0, rgb(46, 52, 64);
        nord1:  #3b4252; dark1, rgb(59, 66, 82);
        nord2:  #434c5e; dark2, rgb(67, 76, 94);
        nord3:  #4c566a; dark3, rgb(76, 86, 106);
        nord4:  #d8dee9; light0, rgb(216, 222, 233);
        nord5:  #e5e9f0; light1, rgb(229, 233, 240);
        nord6:  #eceff4; light2, rgb(236, 239, 244);
        nord7:  #8fbcbb; frost green ;
        nord8:  #88c0d0; frost light blue;
        nord9:  #81a1c1; frost blue;
        nord10: #5e81ac; frost indigo;
        nord11: #bf616a; aurora red;
        nord12: #d08770; aurora orange;
        nord13: #ebcb8b; aurora yellow;
        nord14: #a3be8c; aurora green ;
        nord15: #b48ead; aurora violet;
        """
        
        self.nord_color_table = {
            'red':'bf616a',
            'orange':'d08770',
            'yellow':'ebcb8b',
            'green': 'a3be8c',
            'violet':'b48ead',
            'blue':'5e81ac',
            'r':'bf616a',
            'o':'d08770',
            'y':'ebcb8b',
            'g': 'a3be8c',
            'v':'b48ead',
            'b':'5e81ac',
        }
        self.frost_color_table = {
            'frost green':'8fbcbb',
            'frost light blue':'88c0d0',
            'frost blue':'81a1c1',
            'frost indigo':'5e81ac',
            'fg':'8fbcbb',
            'flb':'88c0d0',
            'fb':'81a1c1',
            'fi':'5e81ac',
        }
        self.mono_color_table = {
            'dark0':'2e3440',
            'dark1':'3b4252',
            'dark2':'434c5e',
            'dark3':'4c566a',
            'light0':'d8dee9',
            'light1':'e5e9f0',
            'light2':'eceff4',
            'd0':'2e3440',
            'd1':'3b4252',
            'd2':'434c5e',
            'd3':'4c566a',
            'l0':'d8dee9',
            'l1':'e5e9f0',
            'l2':'eceff4',
            'white': 'FFFFFF',
            'w': 'FFFFFF',
            'black': '000000',
        }
        
        
        self.all_color_table = {}
        self.all_color_table.update(self.nord_color_table)
        self.all_color_table.update(self.frost_color_table)
        self.all_color_table.update(self.mono_color_table)
        
        self.cmap_table = {
            'mono_red': [self.color(c) for c in ('w','r')],
            'mono_orange': [self.color(c) for c in ('w','o')],
            'mono_yellow': [self.color(c) for c in ('w','y')],
            'mono_green': [self.color(c) for c in ('w','g')],
            'mono_violet': [self.color(c) for c in ('w','v')],
            'mono_blue': [self.color(c) for c in ('w','b')],
            'mono': [self.color(c) for c in ('l2','d0')],
            'rb': [self.color(c) for c in ('b','r')],
            'yg': [self.color(c) for c in ('g','y')],
            'virds': [self.color(c) for c in ('b', 'g', 'y')],
            #'rainbow' : [self.color(c) for c in ('b', 'fb', 'g', 'y', 'o', 'r')],
            'rainbow' : [self.color(c) for c in ('blue', 'frost light blue', 'frost green', 'green', 'yellow', 'orange', 'red')],
            
        }
        for key, val in self.cmap_table.items():
            self.register_cmap('nord_'+key, val)
        pass
    
    def color(self, c):
        if c in self.all_color_table.keys():
            return '#'+self.all_color_table[c]
        
        raise ValueError(f'{c} is not in the color table!!')
    
    def register_cmap(self, name, clist):
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        
        cmap_dis = ListedColormap(clist)
        cmap_seq = LinearSegmentedColormap.from_list(name, clist)
        
        import matplotlib.pyplot as plt
        plt.register_cmap(name + '_dis',   cmap_dis)
        plt.register_cmap(name + '_dis_r', cmap_dis.reversed())
        plt.register_cmap(name,            cmap_seq)
        plt.register_cmap(name + '_r',     cmap_seq.reversed())
    
    
    def get_colors(self, N = 100, name = 'nord_rainbow'):
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap(name, N)
        colors = []
        for idx in range(N):
            colors.append( cmap([float(idx/(N - 1))]) )
        return colors
        
    
nord_color = NordColor()

