import logging
import pandas as pd
from fitxf.math.graph.GraphUtils import GraphUtils
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Pandas import Pandas


class GraphUtilsUnitTest:
    def __init__(self, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def test(self):
        gu = GraphUtils(logger=self.logger)
        MAX_DIST = 999999
        edge_data = [
            {'key': 'plane', 'u': 'Shanghai', 'v': 'Tokyo', 'distance': 10, 'comment': 'Shanghai-Tokyo flight'},
            # duplicate (will not be added), order does not matter
            {'key': 'plane', 'u': 'Tokyo', 'v': 'Shanghai', 'distance': 22, 'comment': 'Tokyo-Shanghai flight'},
            # teleport path Tokyo --> Beijing --> Shanghai shorter distance than plane Tokyo --> Shanghai
            {'key': 'teleport', 'u': 'Tokyo', 'v': 'Beijing', 'distance': 2, 'comment': 'Tokyo-Beijing teleport'},
            {'key': 'plane', 'u': 'Tokyo', 'v': 'Beijing', 'distance': 9, 'comment': 'Tokyo-Beijing plane'},
            {'key': 'teleport', 'u': 'Beijing', 'v': 'Shanghai', 'distance': 1, 'comment': 'Beijing-Shanghai teleport'},
            # Other paths
            {'key': 'ship', 'u': 'Shanghai', 'v': 'Tokyo', 'distance': 100, 'comment': 'Shanghai-Tokyo sea'},
            {'key': 'plane', 'u': 'Moscow', 'v': 'Tokyo', 'distance': 100, 'comment': 'Asia-Russia flight'},
            {'key': 'train', 'u': 'Moscow', 'v': 'Tokyo', 'distance': 10000, 'comment': 'Asia-Russia train'},
            {'key': 'ship', 'u': 'Moscow', 'v': 'Tokyo', 'distance': MAX_DIST, 'comment': 'Asia-Russia sea'},
            {'key': 'plane', 'u': 'Medellin', 'v': 'Antartica', 'distance': 888, 'comment': 'Medellin-Antartica'},
        ]
        G_test = {}
        for directed, exp_total_edges, exp_nodes in [
            (False, 9, ['Antartica', 'Beijing', 'Medellin', 'Moscow', 'Shanghai', 'Tokyo']),
            (True, 10, ['Antartica', 'Beijing', 'Medellin', 'Moscow', 'Shanghai', 'Tokyo']),
        ]:
            G_tmp = gu.create_multi_graph(
                edges = edge_data,
                col_weight = 'distance',
                directed = directed,
            )
            G_test[directed] = G_tmp
            print('-----------------------------------------------------------------------------')
            print('Edges (directed=' + str(directed) + ')')
            [print(i, G_tmp.get_edge_data(u=u, v=v)) for i, (u, v, key) in enumerate(G_tmp.edges)]
            all_edges = list(G_tmp.edges)
            all_edges.sort()
            assert len(G_tmp.edges) == exp_total_edges, \
                'Directed ' + str(directed) + ' Expect ' + str(exp_total_edges) + ' edges, but got ' \
                + str(len(G_tmp.edges))
            print('-----------------------------------------------------------------------------')
            print('Nodes (directed=' + str(directed) + ')')
            print(G_tmp.nodes)
            all_nodes = list(G_tmp.nodes)
            all_nodes.sort()
            assert all_nodes == ['Antartica', 'Beijing', 'Medellin', 'Moscow', 'Shanghai', 'Tokyo'], \
                'Directed ' + str(directed) + ' Nodes not expected ' + str(all_nodes)

        paths_dijkstra = {}
        paths_shortest = {}
        for dir in [True, False]:
            paths_dijkstra[dir] = gu.get_dijkstra_path_all_pairs(G=G_test[dir])
            print('-----------------------------------------------------------------------------')
            print('Dijkstra Paths (directed = ' + str(dir) + ')')
            print(pd.DataFrame.from_records([{'edge': k, 'dijkstra-path': v} for k, v in paths_dijkstra[dir].items()]))

            paths_shortest[dir] = gu.get_shortest_path_all_pairs(G=G_test[dir])
            print('-----------------------------------------------------------------------------')
            print('Shortest Paths (directed = ' + str(dir) + ')')
            print(pd.DataFrame.from_records([{'edge': k, 'shortest-path': v} for k, v in paths_shortest[dir].items()]))

        for dir, edge, exp_best_path in [
            # teleport path for undirected graph from Shanghai-->Beijing-->Tokyo is fastest
            (False, ('Shanghai', 'Tokyo'), ['Shanghai', 'Beijing', 'Tokyo']),
            (False, ('Shanghai', 'Moscow'), ['Shanghai', 'Beijing', 'Tokyo', 'Moscow']),
            # no teleport path for directed graph from Shanghai-->Tokyo
            (True, ('Shanghai', 'Tokyo'), ['Shanghai', 'Tokyo']),
            (True, ('Shanghai', 'Moscow'), None),
        ]:
            observed_edge = paths_dijkstra[dir].get(edge, None)
            assert observed_edge == exp_best_path, \
                'Directed "' + str(dir) + '" Edge ' + str(edge) + ' path ' + str(observed_edge) \
                + ' not expected ' + str(exp_best_path)

        for dir, edge, exp_best_path in [
            (False, ('Shanghai', 'Tokyo'), ['Shanghai', 'Tokyo']),
            (False, ('Shanghai', 'Moscow'), ['Shanghai', 'Tokyo', 'Moscow']),
            (True, ('Shanghai', 'Tokyo'), ['Shanghai', 'Tokyo']),
            (True, ('Shanghai', 'Moscow'), None),
        ]:
            observed_edge = paths_shortest[dir].get(edge, None)
            assert observed_edge == exp_best_path, \
                'Edge ' + str(edge) + ' path ' + str(observed_edge) + ' not expected ' + str(exp_best_path)

        print('-----------------------------------------------------------------------------')
        for dir, source, target, method, exp_path in [
            (False, 'Tokyo', 'Shanghai', 'dijkstra', ['Tokyo', 'Beijing', 'Shanghai']),
            (False, 'Tokyo', 'Shanghai', 'shortest', ['Tokyo', 'Shanghai']),
            (False, 'Tokyo', 'Shanghai', 'simple', ['Tokyo', 'Shanghai']),
            (False, 'Tokyo', 'Antartica', 'dijkstra', None),
            (False, 'Tokyo', 'Antartica', 'shortest', None),
            (False, 'Tokyo', 'Antartica', 'simple', None),
        ]:
            print(str(source) + ' --> ' + str(target))
            paths = gu.get_paths(
                G = G_test[dir],
                source = source,
                target = target,
                method = method,
            )
            print('Best path method "' + str(method) + '" ' + str(source) + '--' + str(target) + ': ' + str(paths))
            best_path = paths[0]['path'] if len(paths)>0 else None
            assert best_path == exp_path, \
                'Best path "' + str(method) + '" ' + str(best_path) + ' not ' + str(exp_path)

        #
        # Search test
        #
        for dir, query_conns, exp_top_keys in [
            (
                    False, [{'u': 'Bangkok', 'v': 'Moscow'}, {'u': 'Tokyo', 'v': 'Shanghai'}],
                    {1: [], 2: ['teleport']}
            ),
            (
                    False, [{'u': 'Bangkok', 'v': 'Moscow'}, {'u': 'Moscow', 'v': 'Shanghai'}],
                    {1: [], 2: [], 3: ['plane', 'teleport']},
            ),
            (
                    False, [{'u': 'Antartica', 'v': 'Medellin'}, {'u': 'Beijing', 'v': 'Shanghai'}],
                    {1: ['plane', 'teleport']},
            ),
        ]:
            res = gu.search_top_keys_for_edges(
                query_edges = query_conns,
                ref_multigraph = G_test[dir],
            )
            self.logger.info('Return search result: ' + str(res))
            top_keys = res['top_keys_by_number_of_edges']
            assert top_keys == exp_top_keys, 'Result top keys ' + str(top_keys) + ' not ' + str(exp_top_keys)

        # gu.draw_graph(G=G_test[False], weight_large_thr=50, agg_weight_by='min')
        self.logger.info('ALL TESTS PASSED')
        return


if __name__ == '__main__':
    Pandas.increase_display()
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    GraphUtilsUnitTest(logger=lgr).test()
    exit(0)
