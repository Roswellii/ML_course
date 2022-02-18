import pydot
# tree= {'属性编号': 5, '属性值': 6.781, 'left': {'属性编号': 12, '属性值': 10.13, 'left': {'属性编号': 5, '属性值': 6.538, 'left': {'属性编号': 9, '属性值': 223.0, 'left': 28.625, 'right': {'属性编号': 5, '属性值': 5.961, 'left': {'属性编号': 11, '属性值': 395.18, 'left': 22.075000000000003, 'right': {'属性编号': 6, '属性值': 41.5, 'left': 21.05, 'right': 19.7}}, 'right': {'属性编号': 0, '属性值': 0.12757, 'left': {'属性编号': 12, '属性值': 7.43, 'left': {'属性编号': 9, '属性值': 281.0, 'left': 25.975, 'right': {'属性编号': 4, '属性值': 0.405, 'left': 22.775, 'right': {'属性编号': 0, '属性值': 0.07896, 'left': 24.375, 'right': 23.32}}}, 'right': {'属性编号': 11, '属性值': 394.72, 'left': {'属性编号': 2, '属性值': 4.05, 'left': 22.45, 'right': 21.5}, 'right': {'属性编号': 0, '属性值': 0.04981, 'left': 22.55, 'right': 23.4}}}, 'right': {'属性编号': 9, '属性值': 277.0, 'left': 25.775, 'right': {'属性编号': 12, '属性值': 6.43, 'left': 25.124999999999996, 'right': 24.071428571428573}}}}}, 'right': {'属性编号': 12, '属性值': 4.69, 'left': 30.683333333333334, 'right': {'属性编号': 7, '属性值': 3.6519, 'left': {'属性编号': 5, '属性值': 6.625, 'left': 30.1, 'right': 27.925}, 'right': {'属性编号': 5, '属性值': 6.619, 'left': 24.619999999999997, 'right': {'属性编号': 12, '属性值': 6.27, 'left': 27.775, 'right': 25.725}}}}}, 'right': {'属性编号': 12, '属性值': 16.03, 'left': {'属性编号': 9, '属性值': 277.0, 'left': {'属性编号': 4, '属性值': 0.464, 'left': {'属性编号': 7, '属性值': 4.429, 'left': 21.26, 'right': 19.775}, 'right': {'属性编号': 4, '属性值': 0.489, 'left': 24.96666666666667, 'right': 22.133333333333336}}, 'right': {'属性编号': 12, '属性值': 12.43, 'left': {'属性编号': 5, '属性值': 5.787, 'left': 18.275, 'right': {'属性编号': 10, '属性值': 17.4, 'left': {'属性编号': 12, '属性值': 11.65, 'left': 22.999999999999996, 'right': 21.475}, 'right': {'属性编号': 6, '属性值': 81.6, 'left': 20.928571428571427, 'right': {'属性编号': 6, '属性值': 85.4, 'left': 18.625, 'right': 20.7}}}}, 'right': {'属性编号': 0, '属性值': 0.14476, 'left': {'属性编号': 7, '属性值': 5.4509, 'left': 21.099999999999998, 'right': 18.875}, 'right': {'属性编号': 7, '属性值': 4.0123, 'left': {'属性编号': 12, '属性值': 14.1, 'left': 19.840000000000003, 'right': {'属性编号': 12, '属性值': 15.12, 'left': 17.1, 'right': 18.775}}, 'right': 17.042857142857144}}}}, 'right': {'属性编号': 0, '属性值': 0.55778, 'left': {'属性编号': 7, '属性值': 1.9669, 'left': 15.957142857142856, 'right': {'属性编号': 4, '属性值': 0.448, 'left': 17.225, 'right': {'属性编号': 4, '属性值': 0.507, 'left': 21.483333333333334, 'right': {'属性编号': 0, '属性值': 0.1712, 'left': {'属性编号': 12, '属性值': 17.58, 'left': 19.0, 'right': 22.15}, 'right': 17.075000000000003}}}}, 'right': {'属性编号': 6, '属性值': 93.8, 'left': 14.98, 'right': {'属性编号': 0, '属性值': 0.98843, 'left': 15.000000000000002, 'right': {'属性编号': 7, '属性值': 1.5257, 'left': {'属性编号': 5, '属性值': 5.403, 'left': 13.55, 'right': 15.1}, 'right': 13.2}}}}}}, 'right': {'属性编号': 5, '属性值': 7.416, 'left': {'属性编号': 12, '属性值': 9.59, 'left': {'属性编号': 7, '属性值': 2.829, 'left': 36.114285714285714, 'right': {'属性编号': 7, '属性值': 3.4217, 'left': 28.449999999999996, 'right': {'属性编号': 6, '属性值': 13.9, 'left': 30.65, 'right': {'属性编号': 6, '属性值': 27.7, 'left': {'属性编号': 0, '属性值': 0.03359, 'left': 34.349999999999994, 'right': 35.95}, 'right': {'属性编号': 9, '属性值': 242.0, 'left': 34.63333333333333, 'right': {'属性编号': 5, '属性值': 7.107, 'left': 30.880000000000003, 'right': 33.199999999999996}}}}}}, 'right': 27.580000000000002}, 'right': {'属性编号': 10, '属性值': 14.7, 'left': {'属性编号': 5, '属性值': 7.61, 'left': 44.725, 'right': {'属性编号': 12, '属性值': 3.7, 'left': 50.0, 'right': 49.325}}, 'right': {'属性编号': 11, '属性值': 385.05, 'left': 47.160000000000004, 'right': {'属性编号': 0, '属性值': 0.08187, 'left': 44.6, 'right': 39.2}}}}}
tree= {'petal-length': {'1_3.3': {'petal-width': {'1_1.8': {'sepal-length': {'1_6.0': 2, '0_6.0': {'sepal-width': {'1_3.2': 1, '0_3.2': 2}}}}, '0_1.8': {'sepal-length': {'1_7.2': 2, '0_7.2': {'sepal-width': {'1_2.6': 1, '0_2.6': 1}}}}}}, '0_3.3': 0}}
class DrawTree:
    graph= None
    def __init__(self):
       self.graph= pydot.Dot(graph_type='digraph')
    def draw(self, parent, child):
        # 父指向子
        edge= pydot.Edge(parent, child)
        if self.graph.get_edge(parent, child)==[] and self.graph.get_edge(child, parent)==[]:
          self.graph.add_edge(edge)

    def visit(self, node, parent= None):
        if isinstance(node, dict):
            lft= node['left']
            rgt= node['right']
            # self.draw(node['fea_idx'], lft['fea_idx'])
            if isinstance(lft, dict):
             self.draw(str(node['fea_idx']) ,str(lft['fea_idx']))
             #self.draw(str(node['fea_idx'])+ '   ' +str(round(node['fea_val'], 2)), str(lft['fea_idx'])+ '  ' +str(round(lft['fea_val'], 2)))
             self.visit(lft)
            else:
               self.draw(str(node['fea_idx']), str(lft))
            if isinstance(rgt, dict):
             self.draw(str(node['fea_idx']), str(rgt['fea_idx']))
            # self.draw(str(node['fea_idx'])+ '   ' +str(round(node['fea_val'], 2)), str(rgt['fea_idx'])+ '  ' +str(round(rgt['fea_val'], 2)))
             self.visit(rgt)
            else:
               self.draw(str(node['fea_idx']), str(rgt))


        # print(lft)
        # for k, v in node.items():
        #     if isinstance(v, dict): # 如果当前节点是字典
        #         if parent: # 非根节点才画
        #             self.draw(parent, k)
        #         self.visit(v, k) # 访问当前节点
    def start(self, dic):
        self.visit(dic)
        self.graph.write_png('tree.png')
drawtree= DrawTree()
drawtree.start(tree)
