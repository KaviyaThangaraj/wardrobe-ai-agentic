from langgraph.graph import StateGraph, END

from src.graph.WardrobeState import WardrobeState
from src.graph.nodes.ingestion_node import analyse_and_store_wardrobe, analyse_and_store_profile
from src.graph.nodes.router_node import router_node
from src.graph.nodes.stylist_node import stylist_node


def route_intent(state: WardrobeState) -> str:
    if state.get("error"):
        return "end"
    return state["intent"]


def response_node(state: WardrobeState) -> WardrobeState:
    if state.get("error"):
        return {**state, "response": f"Something went wrong: {state['error']}"}
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(WardrobeState)

    # register nodes
    graph.add_node("router", router_node)
    graph.add_node("ingestion", analyse_and_store_wardrobe)
    graph.add_node("profile", analyse_and_store_profile)
    graph.add_node("stylist", stylist_node)
    graph.add_node("response", response_node)

    # entry point
    graph.set_entry_point("router")

    # conditional edges from router
    graph.add_conditional_edges(
        "router",
        route_intent,
        {
            "upload_wardrobe": "ingestion",
            "upload_profile":  "profile",
            "suggest":         "stylist",
            "end":             "response",
        }
    )

    # all nodes converge to response then END
    graph.add_edge("ingestion", "response")
    graph.add_edge("profile",   "response")
    graph.add_edge("stylist",   "response")
    graph.add_edge("response",  END)

    return graph.compile()


wardrobe_graph = build_graph()
