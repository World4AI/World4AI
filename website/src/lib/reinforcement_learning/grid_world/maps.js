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
            distance: 1,
        },
        {
            r: 0,
            c: 1,
            type: "floor",
            reward: -1,
            distance: 0.9,
        },
        {
            r: 0,
            c: 2,
            type: "floor",
            reward: -1,
            distance: 0.8,
        },
        {
            r: 0,
            c: 3,
            type: "floor",
            reward: -1,
            distance: 0.7
        },
        {
            r: 0,
            c: 4,
            type: "floor",
            reward: -1,
            distance: 0.8
        },
        {
            r: 1,
            c: 0,
            type: "floor",
            reward: -1,
            distance: 0.9
        },
        {
            r: 1,
            c: 1,
            type: "floor",
            reward: -1,
            distance: 0.8
        },
        {
            r: 1,
            c: 2,
            type: "floor",
            reward: -1,
            distance: 0.7
        },
        {
            r: 1,
            c: 3,
            type: "floor",
            reward: -1,
            distance: 0.6
        },
        {
            r: 1,
            c: 4,
            type: "floor",
            reward: -1,
            distance: 0.7
        },
        {
            r: 2,
            c: 0,
            type: "block",
            reward: 0,
            distance: 1
        },
        {
            r: 2,
            c: 1,
            type: "block",
            reward: 0,
            distance: 1
        },
        {
            r: 2,
            c: 2,
            type: "block",
            reward: 0,
            distance: 1
        },
        {
            r: 2,
            c: 3,
            type: "floor",
            reward: -1,
            distance: 0.45
        },
        {
            r: 2,
            c: 4,
            type: "floor",
            reward: -1,
            distance: 0.5
        },
        {
            r: 3,
            c: 0,
            type: "floor",
            reward: -1,
            distance: 0.1
        },
        {
            r: 3,
            c: 1,
            type: "floor",
            reward: -1,
            distance: 0.2
        },
        {
            r: 3,
            c: 2,
            type: "floor",
            reward: -1,
            distance: 0.3
        },
        {
            r: 3,
            c: 3,
            type: "floor",
            reward: -1,
            distance: 0.4
        },
        {
            r: 3,
            c: 4,
            type: "floor",
            reward: -1,
            distance: 0.45
        },
        {
            r: 4,
            c: 0,
            type: "goal",
            reward: 1,
            distance: 0
        },
        {
            r: 4,
            c: 1,
            type: "floor",
            reward: -1,
            distance: 0.1
        },
        {
            r: 4,
            c: 2,
            type: "floor",
            reward: -1,
            distance: 0.2
        },
        {
            r: 4,
            c: 3,
            type: "floor",
            reward: -1,
            distance: 0.3
        },
        {
            r: 4,
            c: 4,
            type: "floor",
            reward: -1,
            distance: 0.4
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

const gridCorridor = {
    rows: 1,
    columns: 3,
    player: {
        r: 0,
        c: 0
    },
    actions: {
        east: 0,
        west: 1
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
            type: "goal",
            reward: 1,
        },
    ]
}
export { gridMap, pickupGridMap, gridCorridor }
