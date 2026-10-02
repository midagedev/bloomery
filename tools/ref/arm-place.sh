# shellcheck shell=bash
# An arm's placement in the depth runners whose binary takes --place a|gate|bp (depth-ds41.sh,
# depth-glm5next.sh, depth-qwen3moe.sh): `<arm>@place=<word>[,NAME=VALUE...]` runs that arm at --place
# <word>, over the runner's BLOOMERY_GEN_PLACE; the item is the runner's, never a variable the binary sees.
# What is here, each runner's by its own words and binary, with no model's branch: the item's parse, the
# refusals (one text in every runner: `<runner>: arm '<arm>': <why>`, or `<runner>: <why>` for
# BLOOMERY_GEN_PLACE), the word against the mode and the timing card, and after an arm in the two-card mode
# its load record's cards against its placement. The two-card witness (Xid, the 3090's cap, a lost card) is
# timing-card.sh's, which this file calls. Sourced by the runner before it parses its arms; place_check and
# place_arm_cards need timing-card.sh sourced first.
#
# The runner sets, before it calls a function here:
#   PLACE_RUNNER     the runner's name, which opens every message here
#   PLACE_BIN        its binary's name, for the messages
#   PLACE_WORDS      the place words the runner runs its binary at, space-separated (a gate bp)
#   PLACE_LOAD_KIND  (place_arm_cards, optional) the records.py kind of the binary's load record, default load
#   PLACE_LOAD_BIN   (place_arm_cards, optional) the binary whose checked-in schema reads it, default
#                    records.py's (generate_ds41)
#
# The words: a is plan (a), on the card named A6000 (workstation::plan_a); gate the gate plan, on the 3090
# (plan_gate); bp plan (b′), plan (a) on the A6000 with the 3090 its expert tier (plan_bp). One card (no
# BLOOMERY_TIMING_CARDS) times the timing card alone, so a needs the A6000 as the timing card, gate the 3090,
# and bp is refused. The two-card mode (BLOOMERY_TIMING_CARDS=a6000+3090) shows both cards to every arm: bp
# loads both, a loads the A6000 and leaves the 3090 idle (place_arm_cards holds each a arm's load record to
# the A6000 alone), and gate, a 3090-only row in the A6000+3090 table, is refused.

# place_refuse <arm or empty> <why>: the refusal, exit 64.
place_refuse() {
  if [ -n "$1" ]; then echo "$PLACE_RUNNER: arm '$1': $2" >&2; else echo "$PLACE_RUNNER: $2" >&2; fi
  exit 64
}
# place_words_text: PLACE_WORDS as prose (`a, gate or bp`).
place_words_text() {
  local -a w
  read -r -a w <<< "$PLACE_WORDS"
  if [ ${#w[@]} -le 1 ]; then
    echo "${w[*]}"
  else
    local IFS=,
    local head="${w[*]:0:${#w[@]}-1}"
    echo "${head//,/, } or ${w[${#w[@]} - 1]}"
  fi
}
# arm_place_split <arm> <NAME=VALUE list>: the list's place= item into ARM_PLACE (empty without one) and the
# rest, every other item as given and in order (an empty one too: the runner's list check names it), into
# ARM_REST. An empty place= value, place given twice and a word outside PLACE_WORDS are refused by name.
arm_place_split() {
  local a=$1 s="$2," e rest='' seen=''
  ARM_PLACE=''
  while [ -n "$s" ]; do
    e=${s%%,*} s=${s#*,}
    case $e in
      place=*)
        [ -z "$ARM_PLACE" ] || place_refuse "$a" "place is given twice"
        ARM_PLACE=${e#place=}
        [ -n "$ARM_PLACE" ] || place_refuse "$a" "place= names no placement ($(place_words_text))"
        case " $PLACE_WORDS " in
          *" $ARM_PLACE "*) ;;
          *) place_refuse "$a" "place=$ARM_PLACE: $PLACE_BIN takes --place $(place_words_text) in this runner" ;;
        esac
        ;;
      *) rest+="${seen:+,}$e" seen=1 ;;
    esac
  done
  # shellcheck disable=SC2034 # read by the runner that sources this file
  ARM_REST=$rest
}
# arm_has_place <NAME=VALUE list>: whether the list holds a place= item.
arm_has_place() { [[ ,$1 == *,place=* ]]; }
# place_ref_refuse <arm> <engine>: place= on a reference engine's arm, refused by name.
place_ref_refuse() {
  place_refuse "$1" "place= sets $PLACE_BIN's --place, and $2 is a reference engine's arm, which takes no placement of ours"
}
# place_check <arm or empty> <word>: the word against the mode and the timing card (the header's words), an
# arm's or BLOOMERY_GEN_PLACE's (arm empty); each refusal by name, exit 64. Run before the lease and before a
# dry run's lines.
place_check() {
  local a=$1 p=$2 what
  if [ -n "$a" ]; then what="place=$p"; else what="BLOOMERY_GEN_PLACE=$p"; fi
  case " $PLACE_WORDS " in
    *" $p "*) ;;
    *) place_refuse "$a" "$what: $PLACE_BIN takes --place $(place_words_text) in this runner" ;;
  esac
  if [ -n "$TIMING_CARDS" ]; then
    case $p in
      a | bp) return 0 ;;
      gate) place_refuse "$a" "$what is the gate plan, which loads the 3090 alone; the two-card mode times a (plan (a) on the A6000, the 3090 idle) or bp (plan (b′), both cards)" ;;
      *) place_refuse "$a" "$what has no two-card placement (a: the A6000, the 3090 idle; bp: both cards)" ;;
    esac
  fi
  case $p in
    bp) place_refuse "$a" "$what is plan (b′), which loads on both cards (the A6000 and its 3090 expert tier); it runs in the two-card mode, BLOOMERY_TIMING_CARDS=a6000+3090" ;;
    a)
      [ -n "$GPU_A6000" ] || place_refuse "$a" "$what is plan (a), which loads on the A6000, and tools/ref/cards.sh resolved no A6000 UUID (${CARDS_ERROR:-no reason given})"
      if [ "$TIMING_GPU" != "$GPU_A6000" ]; then
        if [ -n "$GPU_3090" ] && [ "$TIMING_GPU" = "$GPU_3090" ]; then
          place_refuse "$a" "$what is plan (a), which loads on the A6000, and the timing card is the 3090 (BLOOMERY_TIMING_GPU=$TIMING_GPU): $PLACE_BIN would refuse the arm; run it at gate"
        fi
        place_refuse "$a" "$what is plan (a), which loads on the A6000, and the timing card is $TIMING_GPU, not the A6000 ($GPU_A6000)"
      fi
      ;;
    gate)
      if [ -z "$GPU_3090" ]; then
        place_refuse "$a" "$what is the gate plan, which loads on the 3090, and tools/ref/cards.sh cannot name the 3090 (${CARDS_ERROR:-no reason given}): whether the timing card is the 3090 cannot be told"
      fi
      [ "$TIMING_GPU" = "$GPU_3090" ] || place_refuse "$a" "$what is the gate plan, which loads on the 3090, and the timing card is $TIMING_GPU, not the 3090 ($GPU_3090): name the 3090 in BLOOMERY_TIMING_GPU, or run it at a"
      ;;
  esac
}
# place_arm_cards <engine log> <place> [<cards>]: after one of our arms in the two-card mode (0 at once
# without it), timing-card.sh's witness (timing_cards_arm: an Xid since the last arm, a card lost or off its
# cap), then the cards the arm loaded against its placement: <cards> as given (`[name,name]`, each space
# written `_`, for a binary whose load line is no records.py kind), else the `cards` of the log's first load
# record (records.py, PLACE_LOAD_KIND of PLACE_LOAD_BIN's schema). bp must name the A6000 then the 3090; a the
# A6000 alone (plan (a) with the 3090 idle); a load with no cards fails, and so does any other place. 1 with
# TWOCARD_WHY on a failure; TWOCARD_DEVS the devices, ` + ` between two, for the row.
# shellcheck disable=SC2034 # TWOCARD_WHY and TWOCARD_DEVS are timing-card.sh's, read by the runners
place_arm_cards() {
  local log=$1 p=$2 rec d CARDS err
  local -a recbin=()
  TWOCARD_DEVS=''
  [ -n "$TIMING_CARDS" ] || return 0
  timing_cards_arm "$log" none || return 1
  if [ $# -ge 3 ]; then
    CARDS=$3
  else
    [ -z "${PLACE_LOAD_BIN:-}" ] || recbin=(--bin "$PLACE_LOAD_BIN")
    if ! rec=$(python3 "${BASH_SOURCE[0]%/*}/../bloomery/records.py" sh ${recbin[@]+"${recbin[@]}"} - "CARDS=${PLACE_LOAD_KIND:-load}.cards" <<< "$log" 2> /dev/null); then
      err=$(python3 "${BASH_SOURCE[0]%/*}/../bloomery/records.py" sh ${recbin[@]+"${recbin[@]}"} - "CARDS=${PLACE_LOAD_KIND:-load}.cards" <<< "$log" 2>&1 > /dev/null)
      TWOCARD_WHY="records.py did not read the engine's ${PLACE_LOAD_KIND:-load} record: ${err##*$'\n'}"
      return 1
    fi
    eval "$rec"
  fi
  if [ -z "$CARDS" ]; then
    TWOCARD_WHY="the engine's load record names no cards: which cards it loaded cannot be told"
    return 1
  fi
  d=${CARDS#\[} d=${d%\]} d=${d//_/ }
  case $p in
    bp)
      if [[ $d =~ ^[^,]*A6000[^,]*,[^,]*3090[^,]*$ ]]; then
        TWOCARD_DEVS="${d//,/ + }"
        return 0
      fi
      TWOCARD_WHY="the engine's load record names cards $CARDS, not the A6000 and the 3090: a one-card run in the two-card table"
      ;;
    a)
      if [[ $d =~ ^[^,]*A6000[^,]*$ ]]; then
        TWOCARD_DEVS=$d
        return 0
      fi
      TWOCARD_WHY="the engine's load record names cards $CARDS, and --place a loads the A6000 alone: the 3090 must stay idle"
      ;;
    *) TWOCARD_WHY="--place $p has no two-card check (a: the A6000 alone; bp: the A6000 and the 3090)" ;;
  esac
  return 1
}
# place_line <arm index…>: the [config] and [dry] placements line's text from the runner's ARMS and A_PLACE,
# for a run in which an arm's place= sets one: `<arm> <word>, …`, and what the rest follow.
place_line() {
  local i out=''
  for i in "$@"; do out+="${out:+, }${ARMS[$i]} ${A_PLACE[$i]}"; done
  echo "$out (an arm's place= over BLOOMERY_GEN_PLACE=${PLACE:-unset})"
}
