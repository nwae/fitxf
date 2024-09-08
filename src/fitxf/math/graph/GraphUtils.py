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
            # only applicable for "simple" path method
            agg_weight_by: str = 'min',
    ) -> list[dict]:
        assert method in ['simple', 'dijkstra', 'shortest']
        func = nx.dijkstra_path if method in ['dijkstra'] else (
            nx.shortest_path if method in ['shortest'] else nx.shortest_simple_paths
        )
        if method in ['simple']:
            # "simple" method cannot work with multigraph
            G__ = self.convert_multigraph_to_simple_graph(G=G, agg_weight_by=agg_weight_by)
            self.logger.info('Converted graph to non-multigraph for get paths method "' + str(method) + '"')
        else:
            G__ = G
        try:
            nodes_traversed_paths = func(
                G = G__,
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
            nodes_traversed_weight = 0
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

                if agg_weight_by == 'min':
                    arg_best_weight = np.argmin([d['weight'] for k, d in ep_edges])
                else:
                    arg_best_weight = np.argmax([d['weight'] for k, d in ep_edges])

                best_key, best_edge = ep_edges[arg_best_weight]
                best_leg = {
                    'leg_number': i_leg,
                    'leg_u': leg_u, 'leg_v': leg_v,
                    'leg_weight': best_edge['weight'],
                    'leg_key': best_key,
                    'leg_total': len(nodes_traversed_path) - 1,
                }
                nodes_traversed_weight += best_edge['weight']
                best_legs.append(best_leg)
            paths_by_method.append({
                'path': nodes_traversed_path,
                'legs': best_legs,
                'weight_total': nodes_traversed_weight,
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
    # TODO not yet optimized mathematically
    def search_top_keys_for_edges(
            self,
            query_edges: list[dict],
            ref_multigraph: nx.Graph,
            # permitted values "simple", "dijkstra", "shortest"
            path_method = 'dijkstra',
            # only applicable for "simple" path method
            path_agg_weight_by: str = 'min',
            query_col_u = 'u',
            query_col_v = 'v',
            # query_col_key = 'key',
            query_col_weight = 'weight',
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
            w_ref = conn.get(query_col_weight, 0)
            edge = (u, v)
            res = self.get_paths(
                G = multi_graph,
                source = u,
                target = v,
                method = path_method,
                agg_weight_by = path_agg_weight_by,
            )
            self.logger.debug(
                'Query edge #' + str(i) + ' method ' + str(path_method) + ', best paths for edge ' + str(edge)
                + ': ' + str(res)
            )
            if len(res) > 0:
                # if reference weight exists, take path with closest weight
                if path_agg_weight_by == 'min':
                    i_best = np.argmin([abs(d['weight_total'] - w_ref) for d in res])
                else:
                    i_best = np.argmax([abs(d['weight_total'] - w_ref) for d in res])
                self.logger.debug('Best path for method ' + str(path_method) + ': ' + str(res[i_best]))
                best_path_uv = res[i_best]['path']
                best_legs_uv = res[i_best]['legs']
                best_weight_total_uv = res[i_best]['weight_total']
            else:
                best_path_uv = None
                best_legs_uv = None
                best_weight_total_uv = None
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
            'Coverage = ' + str(coverage) + ', path most suitable distances ' + str(query_edges_best_paths)
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
            self.logger.info(
                'Converted type "' + str(type(G)) + '" to type "' + str(G_simple) + '"'
            )
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


if __name__ == '__main__':
    Pandas.increase_display()
    lgr = Logging.get_default_logger(log_level=logging.DEBUG, propagate=False)
    gu = GraphUtils(logger=lgr)
    G = gu.create_multi_graph(
        edges = [
            {'key': 'plane', 'u': 'Shanghai', 'v': 'Tokyo', 'distance': 10, 'comment': 'Shanghai-Tokyo flight'},
            # duplicate (will not be added), order does not matter
            {'key': 'plane', 'u': 'Tokyo', 'v': 'Shanghai', 'distance': 22, 'comment': 'Tokyo-Shanghai flight'},
        ],
        col_weight = 'distance',
    )
    print(G)
    exit(0)
