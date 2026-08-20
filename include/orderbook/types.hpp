#pragma once
#include<limits>
#include<cstdint>

namespace ob  {
using OrderId = uint64_t;
using Price = int32_t;
using Qty = uint32_t;
using Slot = uint32_t;

inline constexpr Slot INVALID_SLOT = std::numeric_limits<Slot>::max();
enum class Side : uint8_t {Bid = 0, Ask =1};

struct alignas(64) Order {
    OrderId id = 0;
    Price price = 0;
    Qty qty = 0;
    Slot prev = INVALID_SLOT;
    Slot next = INVALID_SLOT;
    uint32_t level = 0;
    Side side = Side::Bid;
    bool active = false;
};
static_assert(sizeof(Order) == 64, "Order must be exactly one cache line");
static_assert(alignof(Order) == 64, "Order must be cache-line aligned");

struct PriceLevel {
    Qty total_qty = 0;
    uint32_t order_count = 0;
    Slot head = INVALID_SLOT;
    Slot tail = INVALID_SLOT;
};
}