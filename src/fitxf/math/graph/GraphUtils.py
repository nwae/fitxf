import logging
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fitxf.math.utils.Logging import Logging
from fitxf.math.utils.Pandas import Pandas


class GraphUtils:

    def __init__(
            self,
            logger = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger()
        return

    def create_multi_graph(
            self,
            # expect compulsory keys in dict "key", "u", "v", "weight"
            edges: list[dict],
            # compulsory columns in each dict
            col_u = 'u',
            col_v = 'v',
            col_key = 'key',
            col_weight = 'weight',
            directed = False,
    ) -> nx.Graph:
        self.logger.debug('Directed graph = ' + str(directed) + '.Edges to create graph: ' + str(edges))
        multi_g = nx.MultiDiGraph() if directed else nx.MultiGraph()

        for i, edge_rec in enumerate(edges):
            self.logger.debug('#' + str(i) + ': ' + str(edge_rec))
            u = edge_rec[col_u]
            v = edge_rec[col_v]
            key = edge_rec[col_key]
            weight = edge_rec[col_weight]
            other_params = {k: v for k, v in edge_rec.items() if k not in [col_u, col_v, col_key, col_weight]}
            edge_key = (u, v, key)
            if multi_g.edges.get(edge_key) is not None:
                self.logger.warning(str(i) + '. Edge exists ' + str(edge_key) + ': ' + str(multi_g.edges.get(edge_key)))
            else:
                self.logger.debug(str(i) + '. New edge ' + str(edge_key))
            # There will be no duplicate edges, just overwritten by the last one
            multi_g.add_edge(
                # For type nx.Graph, order of u, v does not matter, searching for the edge (u, v)
                # or (v, u) will return the same thing
                key = key,
                u_for_edge = u,
                v_for_edge = v,
                # User info
                u = u,
                v = v,
                weight = weight,
                params = other_params,
            )
            self.logger.debug(
                'Check edge symmetry, key ' + str((u, v)) + ' retrieved from graph ' + str(multi_g.edges.get(edge_key))
                + ', opposite key ' + str((v, u)) + ' retrieved as ' + str(multi_g.edges.get(edge_key))
            )
        return multi_g

    def get_neighbors(self, G: nx.Graph, node: str):
        return nx.neighbors(G=G, n=node)

    def get_paths(
            self,
            G: nx.Graph,
            source,
            target,
            # permitted values "simple", "dijkstra", "shortest"
            method = 'dijkstra',
            agg_weight_by: str = 'min',
    ) -> list[dict]:
        assert method in ['simple', 'dijkstra', 'shortest']
        func = nx.dijkstra_path if method in ['dijkstra'] else (
            nx.shortest_path if method in ['shortest'] else nx.shortest_simple_paths
        )
        try:
            nodes_traversed_paths = func(
                G = self.convert_multigraph_to_simple_graph(G=G, agg_weight_by=agg_weight_by),
                source = source,
                target = target,
            )
            if method == 'simple':
                nodes_traversed_paths = list(nodes_traversed_paths)
            else:
                # for "dijkstra", "shortest" will only return 1 path, we convert to list
                nodes_traversed_paths = [nodes_traversed_paths]
            self.logger.debug('Nodes traversed path "' + str(method) + '": ' + str(nodes_traversed_paths))
        except Exception as ex_no_path:
            self.logger.error(
                'Path "' + str(method) + '" from "' + str(source) + '" --> "' + str(target) + '": ' + str(ex_no_path)
            )
            return []

        paths_by_method = []
        for nodes_traversed_path in nodes_traversed_paths:
            best_legs = []
            for i_leg in range(1, len(nodes_traversed_path)):
                leg_u = nodes_traversed_path[i_leg - 1]
                leg_v = nodes_traversed_path[i_leg]
                self.logger.debug('Method "' + str(method) + '" checking leg ' + str((leg_u, leg_v)))
                # if multiple will be by dictionary key: edge, e.g.
                # {
                #    'teleport': {'u': 'Tokyo', 'v': 'Beijing', 'weight': 2},
                #    'plane': {'u': 'Tokyo', 'v': 'Beijing', 'weight': 9}
                #  }
                ep = G.get_edge_data(u=leg_u, v=leg_v)
                self.logger.debug('For leg ' + str((leg_u, leg_v)) + ', edge data ' + str(ep))
                # convert to convenient tuples instead of key: values
                ep_edges = [(k, d) for k, d in ep.items()]
                arg_min_weight = np.argmin([d['weight'] for k, d in ep_edges])
                best_key, best_edge = ep_edges[arg_min_weight]
                best_leg = {
                    'leg_number': i_leg,
                    'leg_u': leg_u, 'leg_v': leg_v,
                    'leg_weight': best_edge['weight'],
                    'leg_key': best_key,
                    'leg_total': len(nodes_traversed_path) - 1,
                }
                best_legs.append(best_leg)
            paths_by_method.append({
                'path': nodes_traversed_path,
                'legs': best_legs,
            })
        return paths_by_method

    def __helper_convert_to_edge_path_dict(
            self,
            paths_dict: dict,
    ) -> dict:
        edge_path_dict = {}
        for start in paths_dict.keys():
            d_dest = paths_dict[start]
            [
                self.logger.debug(str(start) + '-->' + str(dest) + ':' + str(path))
                for dest, path in d_dest.items() if start != dest
            ]
            for dest, path in d_dest.items():
                if start != dest:
                    edge_path_dict[(start, dest)] = path
        return edge_path_dict

    def get_dijkstra_path_all_pairs(
            self,
            G: nx.Graph,
    ) -> dict:
        sp = dict(nx.all_pairs_dijkstra_path(G))
        return self.__helper_convert_to_edge_path_dict(paths_dict=sp)

    def get_shortest_path_all_pairs(
            self,
            G: nx.Graph,
    ) -> dict:
        sp = dict(nx.all_pairs_shortest_path(G))
        return self.__helper_convert_to_edge_path_dict(paths_dict=sp)

    # Given a set of edges, we find the paths traversed
    def search_top_keys_for_edges(
            self,
            query_edges: list[dict],
            ref_multigraph: nx.Graph,
            query_col_u = 'u',
            query_col_v = 'v',
    ):
        multi_graph = ref_multigraph
        self.logger.debug('Ref graph edges: ' + str(multi_graph.edges))
        self.logger.debug('Ref graph nodes: ' + str(multi_graph.nodes))

        all_legs = []
        query_edges_best_paths = {}
        for i, conn in enumerate(query_edges):
            # for each query edge, find best legs
            u = conn[query_col_u]
            v = conn[query_col_v]
            edge = (u, v)
            res = self.get_paths(G=multi_graph, source=u, target=v, method='dijkstra')
            if len(res) > 0:
                best_path_uv = res[0]['path']
                best_legs_uv = res[0]['legs']
            else:
                best_path_uv = None
                best_legs_uv = None
            self.logger.debug('Best path for ' + str((u, v)) + ': ' + str(best_path_uv))
            self.logger.info(
                'Conn #' + str(i) + ' for edge ' + str(edge) + ', best path: ' + str(best_path_uv)
                + ', best legs for ' + str((u, v)) + ': ' + str(best_legs_uv)
            )
            if best_legs_uv is not None:
                all_legs = all_legs + best_legs_uv
            query_edges_best_paths[(u, v)] = best_path_uv

        self.logger.info('Path shortest distances: ' + str(query_edges_best_paths))

        # Sort by shortest path to longest
        df_all_legs = pd.DataFrame.from_records(all_legs)
        df_all_legs.sort_values(
            by = ['leg_total', 'leg_number', 'leg_u', 'leg_v', 'leg_weight'],
            ascending = True,
            inplace = True,
        )
        max_legs_total = np.max(df_all_legs['leg_total'])
        self.logger.info(
            'Query-collections connections, max leg total=' + str(max_legs_total)
            + ':\n' + str(df_all_legs)
        )

        # Top paths by number of edges traversed
        top_keys_by_number_of_edges = {}
        for i in range(max_legs_total):
            condition = df_all_legs['leg_total'] == i+1
            docs_unique = list(set(df_all_legs[condition]['leg_key'].tolist()))
            docs_unique.sort()
            # key is how many edges required
            top_keys_by_number_of_edges[i+1] = docs_unique
        self.logger.info('Top keys by number of edges: ' + str(top_keys_by_number_of_edges))

        # Indicators
        coverage = round(
            np.sum(
                [1 for v in query_edges_best_paths.values() if v is not None]
            ) / len(query_edges_best_paths.keys()), 3
        )
        self.logger.info(
            'Coverage = ' + str(coverage) + ', path shortest distances ' + str(query_edges_best_paths)
        )

        return {
            'top_keys_by_number_of_edges': top_keys_by_number_of_edges,
            'indicators': {
                'coverage': coverage,
                'shortest_distances': query_edges_best_paths,
            },
            'leg_details': df_all_legs,
        }

    def convert_multigraph_to_simple_graph(
            self,
            G: nx.Graph,
            agg_weight_by: str = 'min',
    ):
        if type(G) in [nx.MultiGraph, nx.MultiDiGraph]:
            # convert to non-multi graph to draw
            G_simple = nx.Graph()
            for edge in G.edges:
                u, v, key = edge
                e_data = G.get_edge_data(u=u, v=v)
                weights = [d['weight'] for key, d in e_data.items()]
                w = np.max(weights) if agg_weight_by == 'max' else np.min(weights)
                G_simple.add_edge(u_of_edge=u, v_of_edge=v, weight=w)
            self.logger.info('Converted type "' + str(type(G)) + '" to type "' + str(G_simple) + '"')
        else:
            G_simple = G
        return G_simple

    def draw_graph(
            self,
            G: nx.Graph,
            weight_large_thr: float = 0.5,
            # if multigraph, aggregate weight method
            agg_weight_by: str = 'min',
            draw_node_size: int = 100,
            draw_font_size:int = 16,
            draw_line_width: int = 4,
    ):
        G_simple = self.convert_multigraph_to_simple_graph(G=G, agg_weight_by=agg_weight_by)

        elarge = [(u, v) for (u, v, d) in G_simple.edges(data=True) if d["weight"] > weight_large_thr]
        esmall = [(u, v) for (u, v, d) in G_simple.edges(data=True) if d["weight"] <= weight_large_thr]

        pos = nx.spring_layout(G_simple, seed=7)  # positions for all nodes - seed for reproducibility
        # nodes
        nx.draw_networkx_nodes(G_simple, pos, node_size=draw_node_size)
        # edges
        nx.draw_networkx_edges(
            G_simple, pos, edgelist=elarge, width=draw_line_width
        )
        nx.draw_networkx_edges(
            G_simple, pos, edgelist=esmall, width=draw_line_width, alpha=0.5, edge_color="b", style="dashed"
        )
        # node labels
        nx.draw_networkx_labels(G_simple, pos, font_size=draw_font_size, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G_simple, "weight")
        nx.draw_networkx_edge_labels(G_simple, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return


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
