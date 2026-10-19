// Nombres y tamaños de cada bloque de la observación. La spec es docs/specs/observations.md.
// Los índices se derivan de estos arreglos: no hay números mágicos en observation.ts.

export const GRID_CHANNELS = [
  "land", // 0 (igual que en la v1)
  "own", // 1 (igual que en la v1)
  "other", // 2 (igual que en la v1)
  "highland",
  "mountain",
  "shore",
  "ally",
  "neighbor",
  "top3",
  "fallout",
  "own_struct",
  "enemy_struct",
  "own_sam_cov",
  "own_post_cov",
  "enemy_sam_cov",
  "nuke_target",
  "friendly_ships",
  "hostile_ships",
  "attack_out",
  "attack_in",
] as const;
export const NUM_CHANNELS = GRID_CHANNELS.length;
export const CH = Object.fromEntries(GRID_CHANNELS.map((n, i) => [n, i])) as Record<(typeof GRID_CHANNELS)[number], number>;

/** Canales cuyo valor no cambia durante el episodio (con waterNukes apagado). */
export const STATIC_CHANNELS = ["land", "highland", "mountain", "shore"] as const;

// --- Economía -----------------------------------------------------------------
export const MAX_ECON_ROWS = 48;
export const ECON_ROW_FEATURES = [
  "is_city",
  "is_port",
  "is_factory",
  "level",
  "x",
  "y",
  "under_construction",
  "has_station",
  "dist_centroid",
] as const;

export const BUILDABLE_COST_TYPES = ["city", "port", "factory", "post", "sam", "silo", "warship", "atom", "hydrogen", "mirv"] as const;
export const ECON_SCALARS = [
  "max_troops_log",
  "troop_growth",
  "city_troop_share",
  "troops_fill",
  "gold_log",
  "income_total",
  "income_trade",
  "income_train",
  "income_piracy",
  ...BUILDABLE_COST_TYPES.map((t) => `cost_${t}`),
  ...BUILDABLE_COST_TYPES.map((t) => `afford_${t}`),
  "city_levels",
  "port_levels",
  "factory_levels",
  "stations",
  "trade_ships",
  "trains",
  "trade_partners",
  "embargoes_by_me",
  "embargoes_on_me",
] as const;

// --- Defensa y ataque -----------------------------------------------------------
export const MAX_MIL_ROWS = 96;
export const MIL_ROW_FEATURES = [
  "is_post",
  "is_sam",
  "is_silo",
  "is_warship",
  "is_transport",
  "is_atom",
  "is_hydrogen",
  "is_mirv",
  "is_warhead",
  "own",
  "friendly",
  "other",
  "x",
  "y",
  "level",
  "under_construction",
  "health",
  "ready_slots",
  "range",
  "troops",
  "target_x",
  "target_y",
  "progress",
  "targeted_by_sam",
  "targets_me",
  "dist_centroid",
  "ws_patrolling",
  "ws_retreating",
  "ws_docked",
  "veterancy",
] as const;

export const MAX_ATTACK_ROWS = 16;
export const MAX_ATTACKS_PER_SIDE = 8;
export const ATTACK_ROW_FEATURES = [
  "outgoing",
  "incoming",
  "troops_over_max",
  "troops_over_mine",
  "retreating",
  "border_size",
  "vs_nobody",
  "from_landing",
  "counterpart_in_table",
  "counterpart_row",
] as const;

export const MIL_SCALARS = [
  "posts",
  "sams",
  "silos",
  "silos_ready",
  "sams_ready",
  "struct_sam_covered",
  "warships",
  "warship_health",
  "warships_in_combat",
  "transports",
  "transport_troops",
  "attacks_out_n",
  "attacks_out_troops",
  "attacks_in_n",
  "attacks_in_troops",
  "attack_in_max",
  "nukes_own_flying",
  "atom_inbound",
  "hydrogen_inbound",
  "mirv_inbound",
  "nuke_inbound_progress",
  "fallout_own",
  "fallout_map",
] as const;

// --- Comparación ----------------------------------------------------------------
export const PLAYER_ROWS = 16;
export const MAX_NEIGHBOR_ROWS = 8;
export const TOP_ROWS = 3;
export const ALLY_ROWS = 4;
export const PLAYER_ROW_FEATURES = [
  // tipo
  "is_self",
  "is_human",
  "is_nation",
  "is_bot",
  // tamaño
  "tiles_share",
  "tiles_ratio",
  "troops_fill",
  "troops_log",
  "troops_ratio",
  "max_troops_ratio",
  // oro
  "gold_log",
  "gold_ratio",
  "income_rate",
  // estructuras
  "cities",
  "ports",
  "factories",
  "posts",
  "sams",
  "silos",
  "warships",
  // relación
  "is_neighbor",
  "can_attack",
  "rel_hostile",
  "rel_distrustful",
  "rel_neutral",
  "rel_friendly",
  "is_ally",
  "targets_me",
  // diplomacia
  "req_in",
  "req_out",
  "alliance_left",
  "can_request",
  "embargo_by_me",
  "embargo_on_me",
  // reputación
  "is_traitor",
  "betrayals",
  "is_immune",
  "disconnected",
  "targeted_by_me",
  // ranking
  "rank",
  "is_top1",
  "is_top3",
  "spawn_dist",
  // presión
  "attacks_on_me",
  "my_attack_on_them",
  "alive",
] as const;

export const CMP_SCALARS = [
  "my_rank",
  "leader_share",
  "leader_ratio",
  "top3_share",
  "win_gap",
  "neighbors_n",
  "neighbors_share",
  "neighbors_troops_ratio",
  "hostile_neighbors",
  "strongest_ratio",
  "weakest_ratio",
  "allies_n",
  "allies_share",
  "open_attack_enemies",
] as const;

// --- Generales (los 6 primeros son los de la v1) ----------------------------------
export const GLOBAL_FEATURES = [
  "land_share",
  "troops_fill",
  "gold_log",
  "alive_ratio",
  "time_progress",
  "in_spawn",
  "win_threshold",
  "leader_share",
  "my_rank",
  "alive_log",
  "humans_alive",
  "nations_alive",
  "spawn_immunity",
  "is_traitor",
] as const;
export const NUM_GLOBALS = GLOBAL_FEATURES.length;

/** Bloques cuyos valores pueden ser negativos (razones con tanh, en [-1, 1]). */
export const SIGNED_BLOCKS = ["players", "comparison"] as const;
