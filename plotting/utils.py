def particle_label(pdgId):
    N = abs(pdgId)

    if   N ==   0: return r'$pp$'
    elif N ==   1: return r'$d$'
    elif N ==   2: return r'$u$'
    elif N ==   3: return r'$s$'
    elif N ==   4: return r'$c$'
    elif N ==   5: return r'$b$'
    elif N ==   6: return r'$top$'
    elif N ==  11: return r'$e$'
    elif N ==  12: return r'$\nu_e$'
    elif N ==  13: return r'$\mu$'
    elif N ==  14: return r'$\nu_\mu$'
    elif N ==  15: return r'$\tau$'
    elif N ==  16: return r'$\nu_\tau$'
    elif N ==  21: return r'$g$'
    elif N ==  22: return r'$\gamma$'
    elif N ==  23: return r'$Z$'
    elif N ==  24: return r'$W$'
    elif N ==  25: return r'$H$'
    elif N == 111: return r'$\pi^0$'
    elif N == 211: return r'$\pi^{\pm}$'
    elif N == 221: return r'$\eta$'
    elif N ==2212: return r'$p$'
    elif N ==2112: return r'$n$'
    elif N == 130: return r'$K^0_L$'
    elif N == 310: return r'$K^0_S$'
    elif N == 311: return r'$K^0$'
    elif N == 321: return r'$K^{\pm}$'
    return f'{n}'

def get_particle_color(pdgId):
    from color import nord_color
    
    pdgId = abs(pdgId)
    
    if pdgId == 0 :
        return nord_color.color('red')
    elif 1 <= pdgId <= 4 :
        return nord_color.color('frost green')
    elif pdgId == 5 : 
        return nord_color.color('violet')
    elif pdgId == 6 : 
        return nord_color.color('blue')
    elif pdgId == 11 or pdgId == 13 : 
        return nord_color.color('yellow')
    elif pdgId == 15 : 
        return nord_color.color('frost blue')
    elif pdgId == 12 or pdgId == 14 or pdgId == 16 : 
        return nord_color.color('light0')
    elif pdgId == 21 : 
        return nord_color.color('light1')
    elif pdgId == 22 : 
        return nord_color.color('yellow')
    elif pdgId == 23 : 
        return nord_color.color('frost blue')
    elif pdgId == 24 : 
        return nord_color.color('orange')
    elif pdgId == 23 : 
        return nord_color.color('yellow')
    else:
        return nord_color.color('frost light blue')
    

import networkx as nx
import random
import numpy as np 

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc - vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def particle_diagram_pos(G, phi_list, root = None, dR = 0.1, center = (0.0, 0.0), rphi = False ):
    
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))
    
    def _particle_diagram_pos(G, _root, dR = 0.1, center = (0.0, 0.0), pos = None, parent = None):
        if pos is None : 
            pos = {_root: center}
        else:
            pos[_root] = center
        
        children = list(G.neighbors(_root))
        
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        
        if len(children) != 0 :
            
            for child in children:
                p = len(nx.shortest_path(G, source = root, target = child)) - 1
                _dR =  p * dR
                phi = phi_list[child]
                if rphi : 
                    c = (_dR, phi)
                else:
                    c = (_dR*np.cos(phi), _dR*np.sin(phi))
                
                pos = _particle_diagram_pos(G, _root = child, dR = dR, center = c, pos = pos, parent = _root )
        return pos
    return _particle_diagram_pos(G, root, dR, center)
        
        
    
























