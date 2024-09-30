import logging
import pandas as pd
import numpy as np
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

        G = gu.create_multi_graph(
            edges = [
                {'key': 'plane', 'u': 'Shanghai', 'v': 'Tokyo', 'distance': 10},
                {'key': 'ship', 'u': 'Shanghai', 'v': 'Tokyo', 'distance': 100},
                {'key': 'plane', 'u': 'Tokyo', 'v': 'Shanghai', 'distance': 22},
                {'key': 'plane', 'u': 'Tokyo', 'v': 'Seoul', 'distance': 5},
                {'key': 'plane', 'u': 'Seoul', 'v': 'Tokyo', 'distance': 6},
                {'key': 'ship', 'u': 'Seoul', 'v': 'Tokyo', 'distance': 60},
            ],
            col_weight = 'distance',
            directed = True,
        )
        for u, v, method, agg, exp_path, exp_weight in [
            ('Shanghai', 'Seoul', 'dijkstra', 'min', ['Shanghai', 'Tokyo', 'Seoul'], 15),
            ('Shanghai', 'Seoul', 'simple', 'min', ['Shanghai', 'Tokyo', 'Seoul'], 15),
            ('Shanghai', 'Seoul', 'simple', 'max', ['Shanghai', 'Tokyo', 'Seoul'], 105),
            # Reverse trip
            ('Seoul', 'Shanghai', 'dijkstra', 'min', ['Seoul', 'Tokyo', 'Shanghai'], 28),
            ('Seoul', 'Shanghai', 'simple', 'min', ['Seoul', 'Tokyo', 'Shanghai'], 28),
            ('Seoul', 'Shanghai', 'simple', 'max', ['Seoul', 'Tokyo', 'Shanghai'], 82),
        ]:
            res = gu.get_paths(G=G, source=u, target=v, method=method, agg_weight_by=agg)
            best_path, best_weight = res[0]['path'], res[0]['weight_total']
            assert best_path == exp_path, \
                'Best path "' + str(method) + '" ' + str(best_path) + ' not ' + str(exp_path)
            assert best_weight == exp_weight, \
                'Best weight "' + str(method) + '" ' + str(best_weight) + ' not ' + str(exp_weight)

        # ---------------------------------------------------------------------------------------------- #

        #
        # Search test
        #
        edge_data_search = [
            {'key': 'plane', 'u': 'Shanghai', 'v': 'Tokyo', 'cost': 10, 'comment': 'Shanghai-Tokyo flight'},
            # duplicate (will not be added), order does not matter
            {'key': 'plane', 'u': 'Tokyo', 'v': 'Shanghai', 'cost': 22, 'comment': 'Tokyo-Shanghai flight'},
            # teleport path Tokyo --> Beijing --> Shanghai shorter distance than plane Tokyo --> Shanghai
            {'key': 'teleport', 'u': 'Tokyo', 'v': 'Beijing', 'cost': 2, 'comment': 'Tokyo-Beijing teleport'},
            {'key': 'teleport-2', 'u': 'Tokyo', 'v': 'Beijing', 'cost': 2, 'comment': 'Tokyo-Beijing teleport'},
            {'key': 'plane', 'u': 'Tokyo', 'v': 'Beijing', 'cost': 9, 'comment': 'Tokyo-Beijing plane'},
            {'key': 'teleport', 'u': 'Beijing', 'v': 'Shanghai', 'cost': 1, 'comment': 'Beijing-Shanghai teleport'},
            # car
            {'key': 'car', 'u': 'Beijing', 'v': 'Moscow', 'cost': 1000000, 'comment': 'Beijing-Moscow car'},
            {'key': 'ev-car', 'u': 'Beijing', 'v': 'Moscow', 'cost': 100000, 'comment': 'Beijing-Moscow EV car'},
            # Other paths
            {'key': 'ship', 'u': 'Shanghai', 'v': 'Xabarovsk', 'cost': 100, 'comment': 'Shanghai-Xabarovsk sea'},
            {'key': 'plane', 'u': 'Xabarovsk', 'v': 'Moscow', 'cost': 3, 'comment': 'Xabarovsk-Moscow plane'},
            {'key': 'plane', 'u': 'Xabarovsk', 'v': 'Tokyo', 'cost': 10, 'comment': 'Asia-Russia flight'},
            {'key': 'train', 'u': 'Xabarovsk', 'v': 'Tokyo', 'cost': 10000, 'comment': 'Asia-Russia train'},
            {'key': 'ship', 'u': 'Xabarovsk', 'v': 'Tokyo', 'cost': MAX_DIST, 'comment': 'Asia-Russia sea'},
            {'key': 'plane', 'u': 'Medellin', 'v': 'Antartica', 'cost': 888, 'comment': 'Medellin-Antartica'},
        ]
        G_search = {}
        for directed in [True, False]:
            G_search[directed] = gu.create_multi_graph(
                edges = edge_data_search,
                col_weight = 'cost',
                directed = directed,
            )
        for i, (dir, query_conns, path_method, exp_top_keys, exp_top_keys_agg_w) in enumerate([
            #
            # Dijkstra test
            #
            (
                    # Undirected test, dijkstra
                    False, [
                        {'u': 'Moscow', 'v': 'Beijing'}, {'u': 'Tokyo', 'v': 'Shanghai'},
                        {'u': 'Medellin', 'v': 'Antartica'}, {'u': 'Vientiane', 'v': 'Bangkok'},
                    ], 'dijkstra',
                    {1.0: ['plane'], 2.0: ['teleport'], 3.0: ['plane', 'teleport'], np.inf: [None]},
                    [
                        {'src_tgt': ('Moscow', 'Beijing'), 'leg_key': 'teleport', '__weight': 0.267},
                        {'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'teleport', '__weight': 1.667},
                        {'src_tgt': ('Moscow', 'Beijing'), 'leg_key': 'plane', '__weight': 7.267},
                        {'src_tgt': ('Medellin', 'Antartica'), 'leg_key': 'plane', '__weight': 888.0},
                    ],
            ),
            (
                    # Directed test, dijkstra
                    True, [
                        {'u': 'Moscow', 'v': 'Beijing'}, {'u': 'Tokyo', 'v': 'Shanghai'},
                        {'u': 'Medellin', 'v': 'Antartica'}, {'u': 'Vientiane', 'v': 'Bangkok'},
                    ], 'dijkstra',
                    {1.0: ['plane'], 2.0: ['teleport'], np.inf: [None]},
                    [
                        {'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'teleport', '__weight': 1.667},
                        {'src_tgt': ('Medellin', 'Antartica'), 'leg_key': 'plane', '__weight': 888.0},
                    ],
            ),
            (
                    # Undirected test, shortest path, by min weight
                    False, [
                        {'u': 'Moscow', 'v': 'Beijing'}, {'u': 'Tokyo', 'v': 'Shanghai'},
                        {'u': 'Medellin', 'v': 'Antartica'}, {'u': 'Vientiane', 'v': 'Bangkok'},
                    ], 'shortest///min',
                    {1.0: ['ev-car', 'plane'], np.inf: [None]},
                    [
                        {'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'plane', '__weight': 22.0},
                        {'src_tgt': ('Medellin', 'Antartica'), 'leg_key': 'plane', '__weight': 888.0},
                        {'src_tgt': ('Moscow', 'Beijing'), 'leg_key': 'ev-car', '__weight': 100000.0},
                    ],
            ),
            (
                    # Undirected test, shortest path, by max weight
                    False, [
                        {'u': 'Moscow', 'v': 'Beijing'}, {'u': 'Tokyo', 'v': 'Shanghai'},
                        {'u': 'Medellin', 'v': 'Antartica'}, {'u': 'Vientiane', 'v': 'Bangkok'},
                    ], 'shortest///max',
                    {1.0: ['car', 'plane'], np.inf: [None]},
                    [
                        {'src_tgt': ('Moscow', 'Beijing'), 'leg_key': 'car', '__weight': 1000000.0},
                        {'src_tgt': ('Medellin', 'Antartica'), 'leg_key': 'plane', '__weight': 888.0},
                        {'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'plane', '__weight': 22.0},
                    ],
            ),
            (
                    # Directed test, shortest path, by max weight
                    True, [
                        {'u': 'Moscow', 'v': 'Beijing'}, {'u': 'Tokyo', 'v': 'Shanghai'},
                        {'u': 'Medellin', 'v': 'Antartica'}, {'u': 'Vientiane', 'v': 'Bangkok'},
                    ], 'shortest///max',
                    {1.0: ['plane'], np.inf: [None]},
                    [
                        {'src_tgt': ('Medellin', 'Antartica'), 'leg_key': 'plane', '__weight': 888.0},
                        {'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'plane', '__weight': 22.0},
                    ],
            ),
            (
                    False, [{'u': 'Bangkok', 'v': 'Moscow'}, {'u': 'Moscow', 'v': 'Shanghai'}], 'dijkstra',
                    {4.0: ['plane', 'teleport'], np.inf: [None]},
                    [
                        {'src_tgt': ('Moscow', 'Shanghai'), 'leg_key': 'teleport', '__weight': 0.312},
                        {'src_tgt': ('Moscow', 'Shanghai'), 'leg_key': 'plane', '__weight': 6.812},
                    ],
            ),
            (
                    False, [{'u': 'Antartica', 'v': 'Medellin'}, {'u': 'Beijing', 'v': 'Shanghai'}], 'dijkstra',
                    {1: ['plane', 'teleport']},
                    [
                        {'src_tgt': ('Beijing', 'Shanghai'), 'leg_key': 'teleport', '__weight': 1.0},
                        {'src_tgt': ('Antartica', 'Medellin'), 'leg_key': 'plane', '__weight': 888.0},
                    ],
            ),
            #
            # Simple test
            #
            (
                    False, [{'u': 'Bangkok', 'v': 'Moscow'}, {'u': 'Tokyo', 'v': 'Shanghai'}],
                    'simple///min',
                    {2: ['teleport'], np.inf: [None]},
                    [{'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'teleport', '__weight': 1.667}],
            ),
            (
                    False, [{'u': 'Bangkok', 'v': 'Moscow'}, {'u': 'Tokyo', 'v': 'Shanghai'}],
                    'simple///max',
                    {4.0: ['car', 'plane', 'ship', 'teleport'], np.inf: [None]},
                    [
                        {'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'car', '__weight': 499999.25},
                        {'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'ship', '__weight': 499998.25},
                        {'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'plane', '__weight': 0.0},
                        {'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'teleport', '__weight': 0.0},
                    ],
            ),
            # Simple graph with given query distance that is closer to 'teleport'
            (
                    False, [{'u': 'Bangkok', 'v': 'Moscow'}, {'u': 'Tokyo', 'v': 'Shanghai', 'cost': 1}],
                    'simple///min',
                    {2: ['teleport'], np.inf: [None]},
                    [{'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'teleport', '__weight': 1.667}],
            ),
            # Simple graph with given query distance that is closer to 'plane'
            (
                    False, [{'u': 'Bangkok', 'v': 'Moscow'}, {'u': 'Tokyo', 'v': 'Shanghai', 'cost': 30}],
                    'simple///min',
                    {1: ['plane'], np.inf: [None]},
                    [{'src_tgt': ('Tokyo', 'Shanghai'), 'leg_key': 'plane', '__weight': 22.0}],
            ),
        ]):
            self.logger.info('Test #' + str(i) + ': ' + str(query_conns))
            if path_method != 'dijkstra':
                path_method, path_agg_wgt_by = path_method.split("///")
            else:
                path_method, path_agg_wgt_by = "dijkstra", "min"

            res = gu.search_top_keys_for_edges(
                query_edges = query_conns,
                ref_multigraph = G_search[dir],
                path_method = path_method,
                # For dijkstra is always "min", so no effect for dijkstra
                path_agg_weight_by = path_agg_wgt_by,
                query_col_u = 'u',
                query_col_v= 'v',
                query_col_weight = 'cost',
            )
            self.logger.info('Return search result:')
            [self.logger.info(str(k) + ': ' + str(v)) for k, v in res.items()]
            top_keys = res['top_keys_by_number_of_edges']
            top_keys_by_agg_weight = res['top_keys_by_aggregated_weight']
            assert top_keys == exp_top_keys, \
                'Result for test #' + str(i) + ' top keys\n' + str(top_keys) + ' not\n' + str(exp_top_keys)
            assert top_keys_by_agg_weight == exp_top_keys_agg_w, \
                'Result for test #' + str(i) + ' top keys by agg weight\n' \
                + str(top_keys_by_agg_weight) + ' not\n' + str(exp_top_keys_agg_w)

        # gu.draw_graph(G=G_test[False], weight_large_thr=50, agg_weight_by='min')
        self.logger.info('ALL TESTS PASSED')
        return


if __name__ == '__main__':
    Pandas.increase_display()
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    GraphUtilsUnitTest(logger=lgr).test()
    exit(0)
