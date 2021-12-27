const gridMap = {
    rows: 5,
    columns: 5,
    player: {
        r: 0,
        c: 0
    },
    actions: {
        north: 0,
        east: 1,
        south: 2,
        west: 3
    },
    cells: [
        {
            r: 0,
            c: 0,
            type: "floor",
            reward: -1,
        },
        {
            r: 0,
            c: 1,
            type: "floor",
            reward: -1,
        },
        {
            r: 0,
            c: 2,
            type: "floor",
            reward: -1
        },
        {
            r: 0,
            c: 3,
            type: "floor",
            reward: -1
        },
        {
            r: 0,
            c: 4,
            type: "floor",
            reward: -1
        },
        {
            r: 1,
            c: 0,
            type: "floor",
            reward: -1
        },
        {
            r: 1,
            c: 1,
            type: "floor",
            reward: -1
        },
        {
            r: 1,
            c: 2,
            type: "floor",
            reward: -1
        },
        {
            r: 1,
            c: 3,
            type: "floor",
            reward: -1
        },
        {
            r: 1,
            c: 4,
            type: "floor",
            reward: -1
        },
        {
            r: 2,
            c: 0,
            type: "block",
            reward: 0
        },
        {
            r: 2,
            c: 1,
            type: "block",
            reward: 0
        },
        {
            r: 2,
            c: 2,
            type: "block",
            reward: 0
        },
        {
            r: 2,
            c: 3,
            type: "floor",
            reward: -1
        },
        {
            r: 2,
            c: 4,
            type: "floor",
            reward: -1
        },
        {
            r: 3,
            c: 0,
            type: "floor",
            reward: -1
        },
        {
            r: 3,
            c: 1,
            type: "floor",
            reward: -1
        },
        {
            r: 3,
            c: 2,
            type: "floor",
            reward: -1
        },
        {
            r: 3,
            c: 3,
            type: "floor",
            reward: -1
        },
        {
            r: 3,
            c: 4,
            type: "floor",
            reward: -1
        },
        {
            r: 4,
            c: 0,
            type: "goal",
            reward: 1
        },
        {
            r: 4,
            c: 1,
            type: "floor",
            reward: -1
        },
        {
            r: 4,
            c: 2,
            type: "floor",
            reward: -1
        },
        {
            r: 4,
            c: 3,
            type: "floor",
            reward: -1
        },
        {
            r: 4,
            c: 4,
            type: "floor",
            reward: -1
        },
    ],
}

const pickupGridMap = {
    rows: 5,
    columns: 5,
    player: {
        r: 0,
        c: 0
    },
    actions: {
        north: 0,
        east: 1,
        south: 2,
        west: 3
    },
    cells: [
        {
            r: 0,
            c: 0,
            type: "floor",
            reward: -1,
        },
        {
            r: 0,
            c: 1,
            type: "floor",
            reward: -1,
        },
        {
            r: 0,
            c: 2,
            type: "floor",
            reward: -1
        },
        {
            r: 0,
            c: 3,
            type: "floor",
            reward: -1
        },
        {
            r: 0,
            c: 4,
            type: "floor",
            reward: -1
        },
        {
            r: 1,
            c: 0,
            type: "floor",
            reward: -1
        },
        {
            r: 1,
            c: 1,
            type: "floor",
            reward: -1
        },
        {
            r: 1,
            c: 2,
            type: "floor",
            reward: -1
        },
        {
            r: 1,
            c: 3,
            type: "floor",
            reward: -1
        },
        {
            r: 1,
            c: 4,
            type: "floor",
            reward: -1
        },
        {
            r: 2,
            c: 0,
            type: "block",
            reward: 0
        },
        {
            r: 2,
            c: 1,
            type: "block",
            reward: 0
        },
        {
            r: 2,
            c: 2,
            type: "block",
            reward: 0
        },
        {
            r: 2,
            c: 3,
            type: "floor",
            reward: -1
        },
        {
            r: 2,
            c: 4,
            type: "floor",
            reward: -1
        },
        {
            r: 3,
            c: 0,
            type: "floor",
            reward: -1
        },
        {
            r: 3,
            c: 1,
            type: "floor",
            reward: -1
        },
        {
            r: 3,
            c: 2,
            type: "floor",
            reward: -1
        },
        {
            r: 3,
            c: 3,
            type: "floor",
            reward: -1
        },
        {
            r: 3,
            c: 4,
            type: "floor",
            reward: -1
        },
        {
            r: 4,
            c: 0,
            type: "goal",
            reward: 1
        },
        {
            r: 4,
            c: 1,
            type: "floor",
            reward: -1
        },
        {
            r: 4,
            c: 2,
            type: "floor",
            reward: -1
        },
        {
            r: 4,
            c: 3,
            type: "floor",
            reward: -1
        },
        {
            r: 4,
            c: 4,
            type: "floor",
            reward: 10
        },
    ],
}

export { gridMap, pickupGridMap }
