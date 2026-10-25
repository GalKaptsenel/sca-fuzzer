# Device controls for violation-261023-072421 (confirmation reps)

Position-controlled device measurements on the Pixel-8 Cortex-X3 (config s24_mte_random.yml,
pinned core 8, view_rotation on, superbatch_size 1). A fixed prefix of 14 genuine variants of
other inputs holds the batch context, and the suspect is swapped at one fixed tail slot. These
extend the controls in VERDICT_072421.md with fresh independent repeats and record the two
remaining optional controls.

## 1. Positive control: userskel_155905 shows the present-on-match polarity

ctrl_pos_tag.py on userskel_155905 (suspect input 5, set 9, --decoy noncanon, reps=3000), three
independent repeats.

| rep | gen@tail | dec@tail (noncanon) | delta tag | delta pos (tail minus head) |
|-----|----------|---------------------|-----------|-----------------------------|
| 1   | 1.00     | 0.00                | -1.00     | -0.00                       |
| 2   | 1.00     | 0.00                | -0.99     | -0.00                       |
| 3   | 1.00     | 0.00                | -1.00     | -0.00                       |

Set 9 is resident at the genuine (matching) tag and cold on the tag mismatch, with no position
component at the fixed slot. This is the present-on-match polarity of TikTag-v1, the opposite
sign of 072421, measured on the same rig. The rig resolves both polarities.

## 2. Granule-305 tracking reconfirm for 072421

ctrl_store_track.py on fuzzer_072421 (suspect 1, set 9, granule 305, --gate-off 0x120,
--tags 3,6,9,12, reps=3000). cell_tag=6, correct tag C=7, base@tail=0.00. Three fresh independent
repeats, each sweeping the sealed access's pointer tag over all 16 deltas at the tail slot. The
suppressed (match) delta is the one where the speculative pointer tag equals the granule's memory
tag T, predicted at (C + T - cell_tag) mod 16 = (1 + T) mod 16.

| memory tag T | predicted match delta | observed dip (rep1 / rep2 / rep3) | residency at match delta | residency at the 13 checked mismatches |
|--------------|-----------------------|-----------------------------------|--------------------------|----------------------------------------|
| 3            | 4                     | 4 / 4 / 4                         | 0.00 / 0.01 / 0.00       | 1.00                                   |
| 6            | 7                     | 7 / 7 / 7                         | 0.00 / 0.00 / 0.00       | 1.00                                   |
| 9            | 10                    | 10 / 10 / 10                      | 0.00 / 0.00 / 0.00       | 1.00                                   |
| 12           | 13                    | 13 / 13 / 13                      | 0.00 / 0.00 / 0.00       | 1.00                                   |

The suppressed delta tracks the retagged granule's memory tag in every row of every repeat
(12 of 12). Set 9 goes cold only where the speculative pointer tag matches granule 305's tag and
is resident at all 13 checked mismatches. These three repeats match the two in
TRACK2D_072421_set9_gran305.txt (reps=2000). The dip sits at the match because 072421 carries the
set on the mismatch (the gain polarity, opposite to section 1).

## 3. Non-MTE (bit-55 sign-extension) control

A bit-55 perturbation is a SignExtSealing seal, and the seal mechanism is fixed when the test
case is generated. MteSealing renders the tag change as an ADDG on the pointer tag field [59:56],
SignExtSealing renders an EOR of bit 55, and no relocation rewrites one slot into the other, so
072421's ADDG tag seal cannot become a bit-55 seal in place. noncanon_variant on an MTE program
forces each eligible seal to a mismatched MTE tag rather than a bit-55 flip. SignExtSealing is
produced only with enable_signext set, which the generator couples to the PAC category, and the
072421 config carries no PAC category. A bit-55 control therefore runs on a separate PAC and
sign-extension program with its own harness. That construction is software only and needs no
kernel-module change. The MTE dependence of 072421 is already carried by the anchor-move control,
where the extra set block follows the sealed speculative load's granule, together with
ENABLE_FAULTY_DEFAULT=0, under which the region label carries no architectural fault and the only
fault source is the MTE tag mismatch.

## 4. Tag-check-off causality

The kernel module exposes no runtime switch for the MTE tag check. The sysfs interface
(/sys/executor) carries enable_pre_run_flush, enable_phr_flush, enable_view_rotation,
enable_bpu_probe, enable_ssbs, measurement_mode, pin_to_core, the reload_* counters, and
read-only register views (tcr_el1, id_aa64isar1_el1, id_aa64isar2_el1, id_aa64pfr1_el1), none of
which writes TCO, TCMA1, or SCTLR_EL1.TCF. The character-device ioctl set (checkout and unload
test case, allocate and free and checkout input, measurement, trace, clear inputs, get test
length, PAC sign and auth and xpac) carries no tag-check command. The module parameters are
mte_skip_exec (dump state and skip execution), mte_tag_verbose (logging), and mte_kasan_selftest.
The tag-check accessors (mte_set_sync, enable_TCMA1_bit, disable_TCO_bit) run at module init
(main.c) and again at the top of every measurement window (measurement.c), where the executor
re-asserts TCF=SYNC, TCMA1=1, and TCO=0 for each run because the kernel restores SCTLR_EL1 and
TCR_EL1 on a context switch. enable_TCO_bit, which would suppress the check, is defined and never
called. A measurement with the tag check off therefore requires editing measurement.c and
rebuilding and reinserting the module. This is the one remaining optional control and it was not
performed.

## Device state

Collection campaign PAUSED (eval_out/collect/PAUSE in place). No fuzzer process was started.
Device 47061FDJH001K3 healthy, executor module loaded, /dev/executor present.
