#pragma once
#include <orderbook/types.hpp>

namespace pipeline {

enum class EventType : uint8_t { Add, Cancel, Modify };

// deliberately trivially-copyable: no owning ptrs, no allocs
// needs to move through lock-free buffer via plain assignment
   // like Order in order book is not heap-allocated but a resident in the arena
struct OrderEvent {
    EventType type = EventType::Add;
    ob::OrderId id = 0;
    ob::Side side = ob::Side::Bid; // for Add only
    ob::Price price = 0; // for Add only
    ob::Qty qty = 0; // for Add (initial) and Modify (new)

    bool operator==(const OrderEvent& other) const {
        return type == other.type && id == other.id && side == other.side && price == other.price && qty == other.qty;
    }
};

} // namespace pipeline