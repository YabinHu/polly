/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/FormattedStream.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <isl/ctx.h>
#include <isl/flow.h>
#include <isl/options.h>
#include <isl/schedule.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include "polly/ScopInfo.h"
#include "ppcg.h"
#include "ppcg_options.h"
// #include "cuda.h"
// #include "opencl.h"
// #include "cpu.h"

using namespace polly;

struct options {
	struct isl_options *isl;
	struct pet_options *pet;
	struct ppcg_options *ppcg;
	char *input;
	char *output;
};

const char *ppcg_version(void);
static void print_version(void)
{
	printf("%s", ppcg_version());
}

#if 0
ISL_ARGS_START(struct options, options_args)
ISL_ARG_CHILD(struct options, isl, "isl", &isl_options_args, "isl options")
ISL_ARG_CHILD(struct options, pet, "pet", &pet_options_args, "pet options")
ISL_ARG_CHILD(struct options, ppcg, NULL, &ppcg_options_args, "ppcg options")
ISL_ARG_STR(struct options, output, 'o', NULL,
	"filename", NULL, "output filename (c and opencl targets)")
ISL_ARG_ARG(struct options, input, "input", NULL)
ISL_ARG_VERSION(print_version)
ISL_ARGS_END

ISL_ARG_DEF(options, struct options, options_args)
#endif

/* Return a pointer to the final path component of "filename" or
 * to "filename" itself if it does not contain any components.
 */
const char *ppcg_base_name(const char *filename)
{
	const char *base;

	base = strrchr(filename, '/');
	if (base)
		return ++base;
	else
		return filename;
}

/* Copy the base name of "input" to "name" and return its length.
 * "name" is not NULL terminated.
 *
 * In particular, remove all leading directory components and
 * the final extension, if any.
 */
int ppcg_extract_base_name(char *name, const char *input)
{
	const char *base;
	const char *ext;
	int len;

	base = ppcg_base_name(input);
	ext = strrchr(base, '.');
	len = ext ? ext - base : strlen(base);

	memcpy(name, base, len);

	return len;
}

/* Is "stmt" not a kill statement?
 */
static int is_not_kill(struct pet_stmt *stmt)
{
	return !pet_stmt_is_kill(stmt);
}

/* Collect the iteration domains of the statements in "scop" that
 * satisfy "pred".
 */
static __isl_give isl_union_set *collect_domains(struct pet_scop *scop,
	int (*pred)(struct pet_stmt *stmt))
{
	int i;
	isl_set *domain_i;
	isl_union_set *domain;

	if (!scop)
		return NULL;

	domain = isl_union_set_empty(isl_set_get_space(scop->context));

	for (i = 0; i < scop->n_stmt; ++i) {
		struct pet_stmt *stmt = scop->stmts[i];

		if (!pred(stmt))
			continue;

		if (stmt->n_arg > 0)
			isl_die(isl_union_set_get_ctx(domain),
				isl_error_unsupported,
				"data dependent conditions not supported",
				return isl_union_set_free(domain));

		domain_i = isl_set_copy(scop->stmts[i]->domain);
		domain = isl_union_set_add_set(domain, domain_i);
	}

	return domain;
}

/* Collect the iteration domains of the statements in "scop",
 * skipping kill statements.
 */
static __isl_give isl_union_set *collect_non_kill_domains(struct pet_scop *scop)
{
	return collect_domains(scop, &is_not_kill);
}

/* This function is used as a callback to pet_expr_foreach_call_expr
 * to detect if there is any call expression in the input expression.
 * Assign the value 1 to the integer that "user" points to and
 * abort the search since we have found what we were looking for.
 */
static int set_has_call(__isl_keep pet_expr *expr, void *user)
{
	int *has_call = (int *)user;

	*has_call = 1;

	return -1;
}

/* Does "expr" contain any call expressions?
 */
static int expr_has_call(__isl_keep pet_expr *expr)
{
	int has_call = 0;

	if (pet_expr_foreach_call_expr(expr, &set_has_call, &has_call) < 0 &&
	    !has_call)
		return -1;

	return has_call;
}

/* This function is a callback for pet_tree_foreach_expr.
 * If "expr" contains any call (sub)expressions, then set *has_call
 * and abort the search.
 */
static int check_call(__isl_keep pet_expr *expr, void *user)
{
	int *has_call = (int *)user;

	if (expr_has_call(expr))
		*has_call = 1;

	return *has_call ? -1 : 0;
}

/* Does "stmt" contain any call expressions?
 */
#if 0
static int has_call(struct pet_stmt *stmt)
{
	int has_call = 0;

	if (pet_tree_foreach_expr(stmt->body, &check_call, &has_call) < 0 &&
	    !has_call)
		return -1;

	return has_call;
}
#endif

static bool hasCall(ScopStmt *Stmt) {
  BasicBlock *BB = Stmt->getBasicBlock();
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    Instruction *Inst = &(*I);
    if (CallInst *CI = dyn_cast<CallInst>(Inst))
      return true;
  }

  return false;
}

/* Collect the iteration domains of the statements in "scop"
 * that contain a call expression.
 */
#if 0
static __isl_give isl_union_set *collect_call_domains(struct pet_scop *scop)
{
	return collect_domains(scop, &has_call);
}
#endif

static __isl_give isl_union_set *collect_call_domains(Scop *S) {
  isl_union_set *Domain;
  isl_set *Domain_i;

  if (!S)
    return NULL;

  isl_set *Context = S->getContext();
  Domain = isl_union_set_empty(isl_set_get_space(Context));

  for (ScopStmt *Stmt : *S)
    if (hasCall(Stmt)) {
      Domain_i = Stmt->getDomain();
      Domain = isl_union_set_add_set(Domain, Domain_i);
    }

  isl_set_free(Context);
  return Domain;
}

/* Given a union of "tagged" access relations of the form
 *
 *	[S_i[...] -> R_j[]] -> A_k[...]
 *
 * project out the "tags" (R_j[]).
 * That is, return a union of relations of the form
 *
 *	S_i[...] -> A_k[...]
 */
static __isl_give isl_union_map *project_out_tags(
	__isl_take isl_union_map *umap)
{
	isl_union_map *proj;

	proj = isl_union_map_universe(isl_union_map_copy(umap));
	proj = isl_union_set_unwrap(isl_union_map_domain(proj));
	proj = isl_union_map_domain_map(proj);

	umap = isl_union_map_apply_domain(umap, proj);

	return umap;
}

/* Construct a relation from the iteration domains to tagged iteration
 * domains with as range the reference tags that appear
 * in any of the reads, writes or kills.
 * Store the result in ps->tagger.
 *
 * For example, if the statement with iteration space S[i,j]
 * contains two array references R_1[] and R_2[], then ps->tagger will contain
 *
 *	{ S[i,j] -> [S[i,j] -> R_1[]]; S[i,j] -> [S[i,j] -> R_2[]] }
 */
static void compute_tagger(struct ppcg_scop *ps)
{
	isl_union_map *tagged, *tagger;

	tagged = isl_union_map_copy(ps->tagged_reads);
	tagged = isl_union_map_union(tagged,
				isl_union_map_copy(ps->tagged_may_writes));
	tagged = isl_union_map_union(tagged,
				isl_union_map_copy(ps->tagged_must_kills));

	tagger = isl_union_map_universe(tagged);
	tagger = isl_union_set_unwrap(isl_union_map_domain(tagger));
	tagger = isl_union_map_reverse(isl_union_map_domain_map(tagger));

	ps->tagger = tagger;
}

/* Compute the live out accesses, i.e., the writes that are
 * potentially not killed by any kills or any other writes, and
 * store them in ps->live_out.
 *
 * We compute the "dependence" of any "kill" (an explicit kill
 * or a must write) on any may write.
 * The may writes with a "depending" kill are definitely killed.
 * The remaining may writes can potentially be live out.
 */
static void compute_live_out(struct ppcg_scop *ps)
{
	isl_union_map *tagger;
	isl_union_map *schedule;
	isl_union_map *empty;
	isl_union_map *kills;
	isl_union_map *exposed;
	isl_union_map *covering;

	tagger = isl_union_map_copy(ps->tagger);
	schedule = isl_union_map_copy(ps->schedule);
	schedule = isl_union_map_apply_domain(schedule,
					isl_union_map_copy(tagger));
	empty = isl_union_map_empty(isl_union_set_get_space(ps->domain));
	kills = isl_union_map_union(isl_union_map_copy(ps->tagged_must_writes),
				    isl_union_map_copy(ps->tagged_must_kills));
	isl_union_map_compute_flow(kills, empty,
				isl_union_map_copy(ps->tagged_may_writes),
				schedule, NULL, &covering, NULL, NULL);
	exposed = isl_union_map_copy(ps->tagged_may_writes);
	exposed = isl_union_map_subtract_domain(exposed,
				isl_union_map_domain(covering));
	exposed = isl_union_map_apply_range(tagger, exposed);
	ps->live_out = exposed;
}

/* Compute the flow dependences and the live_in accesses and store
 * the results in ps->dep_flow and ps->live_in.
 * A copy of the flow dependences, tagged with the reference tags
 * is stored in ps->tagged_dep_flow.
 *
 * We first compute ps->tagged_dep_flow, i.e., the tagged flow dependences
 * and then project out the tags.
 */
static void compute_tagged_flow_dep(struct ppcg_scop *ps)
{
	isl_union_map *tagger;
	isl_union_map *schedule;
	isl_union_map *may_flow;
	isl_union_map *live_in, *may_live_in;

	tagger = isl_union_map_copy(ps->tagger);
	schedule = isl_union_map_copy(ps->schedule);
	schedule = isl_union_map_apply_domain(schedule, tagger);
	isl_union_map_compute_flow(isl_union_map_copy(ps->tagged_reads),
				isl_union_map_copy(ps->tagged_must_writes),
				isl_union_map_copy(ps->tagged_may_writes),
				schedule, &ps->tagged_dep_flow, &may_flow,
				&live_in, &may_live_in);
	ps->tagged_dep_flow = isl_union_map_union(ps->tagged_dep_flow,
							may_flow);
	ps->dep_flow = isl_union_map_copy(ps->tagged_dep_flow);
	ps->dep_flow = isl_union_map_zip(ps->dep_flow);
	ps->dep_flow = isl_union_set_unwrap(isl_union_map_domain(ps->dep_flow));
	live_in = isl_union_map_union(live_in, may_live_in);
	ps->live_in = project_out_tags(live_in);
}

/* Compute the order dependences that prevent the potential live ranges
 * from overlapping.
 * "before" contains all pairs of statement iterations where
 * the first is executed before the second according to the original schedule.
 *
 * In particular, construct a union of relations
 *
 *	[R[...] -> R_1[]] -> [W[...] -> R_2[]]
 *
 * where [R[...] -> R_1[]] is the range of one or more live ranges
 * (i.e., a read) and [W[...] -> R_2[]] is the domain of one or more
 * live ranges (i.e., a write).  Moreover, the read and the write
 * access the same memory element and the read occurs before the write
 * in the original schedule.
 * The scheduler allows some of these dependences to be violated, provided
 * the adjacent live ranges are all local (i.e., their domain and range
 * are mapped to the same point by the current schedule band).
 *
 * Note that if a live range is not local, then we need to make
 * sure it does not overlap with _any_ other live range, and not
 * just with the "previous" and/or the "next" live range.
 * We therefore add order dependences between reads and
 * _any_ later potential write.
 *
 * We also need to be careful about writes without a corresponding read.
 * They are already prevented from moving past non-local preceding
 * intervals, but we also need to prevent them from moving past non-local
 * following intervals.  We therefore also add order dependences from
 * potential writes that do not appear in any intervals
 * to all later potential writes.
 * Note that dead code elimination should have removed most of these
 * dead writes, but the dead code elimination may not remove all dead writes,
 * so we need to consider them to be safe.
 */
static void compute_order_dependences(struct ppcg_scop *ps,
	__isl_take isl_union_map *before)
{
	isl_union_map *reads;
	isl_union_map *shared_access;
	isl_union_set *matched;
	isl_union_map *unmatched;
	isl_union_set *domain;

	reads = isl_union_map_copy(ps->tagged_reads);
	matched = isl_union_map_domain(isl_union_map_copy(ps->tagged_dep_flow));
	unmatched = isl_union_map_copy(ps->tagged_may_writes);
	unmatched = isl_union_map_subtract_domain(unmatched, matched);
	reads = isl_union_map_union(reads, unmatched);
	shared_access = isl_union_map_copy(ps->tagged_may_writes);
	shared_access = isl_union_map_reverse(shared_access);
	shared_access = isl_union_map_apply_range(reads, shared_access);
	shared_access = isl_union_map_zip(shared_access);
	shared_access = isl_union_map_intersect_domain(shared_access,
						isl_union_map_wrap(before));
	domain = isl_union_map_domain(isl_union_map_copy(shared_access));
	shared_access = isl_union_map_zip(shared_access);
	ps->dep_order = isl_union_set_unwrap(domain);
	ps->tagged_dep_order = shared_access;
}

/* Compute the external false dependences of the program represented by "scop"
 * in case live range reordering is allowed.
 * "before" contains all pairs of statement iterations where
 * the first is executed before the second according to the original schedule.
 *
 * The anti-dependences are already taken care of by the order dependences.
 * The external false dependences are only used to ensure that live-in and
 * live-out data is not overwritten by any writes inside the scop.
 *
 * In particular, the reads from live-in data need to precede any
 * later write to the same memory element.
 * As to live-out data, the last writes need to remain the last writes.
 * That is, any earlier write in the original schedule needs to precede
 * the last write to the same memory element in the computed schedule.
 * The possible last writes have been computed by compute_live_out.
 * They may include kills, but if the last access is a kill,
 * then the corresponding dependences will effectively be ignored
 * since we do not schedule any kill statements.
 *
 * Note that the set of live-in and live-out accesses may be
 * an overapproximation.  There may therefore be potential writes
 * before a live-in access and after a live-out access.
 */
static void compute_external_false_dependences(struct ppcg_scop *ps,
	__isl_take isl_union_map *before)
{
	isl_union_map *shared_access;
	isl_union_map *exposed;
	isl_union_map *live_in;

	exposed = isl_union_map_copy(ps->live_out);

	exposed = isl_union_map_reverse(exposed);
	shared_access = isl_union_map_copy(ps->may_writes);
	shared_access = isl_union_map_apply_range(shared_access, exposed);

	ps->dep_external = shared_access;

	live_in = isl_union_map_apply_range(isl_union_map_copy(ps->live_in),
		    isl_union_map_reverse(isl_union_map_copy(ps->may_writes)));

	ps->dep_external = isl_union_map_union(ps->dep_external, live_in);
	ps->dep_external = isl_union_map_intersect(ps->dep_external, before);
}

/* Compute the dependences of the program represented by "scop"
 * in case live range reordering is allowed.
 *
 * We compute the actual live ranges and the corresponding order
 * false dependences.
 */
static void compute_live_range_reordering_dependences(struct ppcg_scop *ps)
{
	isl_union_map *before;

	before = isl_union_map_lex_lt_union_map(
			isl_union_map_copy(ps->schedule),
			isl_union_map_copy(ps->schedule));

	compute_tagged_flow_dep(ps);
	compute_order_dependences(ps, isl_union_map_copy(before));
	compute_external_false_dependences(ps, before);
}

/* Compute the potential flow dependences and the potential live in
 * accesses.
 */
static void compute_flow_dep(struct ppcg_scop *ps)
{
	isl_union_map *may_flow;
	isl_union_map *may_live_in;

	isl_union_map_compute_flow(isl_union_map_copy(ps->reads),
				isl_union_map_copy(ps->must_writes),
				isl_union_map_copy(ps->may_writes),
				isl_union_map_copy(ps->schedule),
				&ps->dep_flow, &may_flow,
				&ps->live_in, &may_live_in);

	ps->dep_flow = isl_union_map_union(ps->dep_flow, may_flow);
	ps->live_in = isl_union_map_union(ps->live_in, may_live_in);
}

/* Compute the dependences of the program represented by "scop".
 * Store the computed potential flow dependences
 * in scop->dep_flow and the reads with potentially no corresponding writes in
 * scop->live_in.
 * Store the potential live out accesses in scop->live_out.
 * Store the potential false (anti and output) dependences in scop->dep_false.
 *
 * If live range reordering is allowed, then we compute a separate
 * set of order dependences and a set of external false dependences
 * in compute_live_range_reordering_dependences.
 */
static void compute_dependences(struct ppcg_scop *scop)
{
	isl_union_map *dep1, *dep2;
	isl_union_map *may_source;

	if (!scop)
		return;

	compute_live_out(scop);

	if (scop->options->live_range_reordering)
		compute_live_range_reordering_dependences(scop);
	else if (scop->options->target != PPCG_TARGET_C)
		compute_tagged_flow_dep(scop);
	else
		compute_flow_dep(scop);

	may_source = isl_union_map_union(isl_union_map_copy(scop->may_writes),
					isl_union_map_copy(scop->reads));
	isl_union_map_compute_flow(isl_union_map_copy(scop->may_writes),
				isl_union_map_copy(scop->must_writes),
				may_source, isl_union_map_copy(scop->schedule),
				&dep1, &dep2, NULL, NULL);

	scop->dep_false = isl_union_map_union(dep1, dep2);
	scop->dep_false = isl_union_map_coalesce(scop->dep_false);
}

/* Eliminate dead code from ps->domain.
 *
 * In particular, intersect ps->domain with the (parts of) iteration
 * domains that are needed to produce the output or for statement
 * iterations that call functions.
 *
 * We start with the iteration domains that call functions
 * and the set of iterations that last write to an array
 * (except those that are later killed).
 *
 * Then we add those statement iterations that produce
 * something needed by the "live" statements iterations.
 * We keep doing this until no more statement iterations can be added.
 * To ensure that the procedure terminates, we compute the affine
 * hull of the live iterations (bounded to the original iteration
 * domains) each time we have added extra iterations.
 */
static void eliminate_dead_code(struct ppcg_scop *ps)
{
	isl_union_set *live;
	isl_union_map *dep;

	live = isl_union_map_domain(isl_union_map_copy(ps->live_out));
	if (!isl_union_set_is_empty(ps->call)) {
		live = isl_union_set_union(live, isl_union_set_copy(ps->call));
		live = isl_union_set_coalesce(live);
	}

	dep = isl_union_map_copy(ps->dep_flow);
	dep = isl_union_map_reverse(dep);

	for (;;) {
		isl_union_set *extra;

		extra = isl_union_set_apply(isl_union_set_copy(live),
					    isl_union_map_copy(dep));
		if (isl_union_set_is_subset(extra, live)) {
			isl_union_set_free(extra);
			break;
		}

		live = isl_union_set_union(live, extra);
		live = isl_union_set_affine_hull(live);
		live = isl_union_set_intersect(live,
					    isl_union_set_copy(ps->domain));
	}

	isl_union_map_free(dep);

	ps->domain = isl_union_set_intersect(ps->domain, live);
}

/* Intersect "set" with the set described by "str", taking the NULL
 * string to represent the universal set.
 */
static __isl_give isl_set *set_intersect_str(__isl_take isl_set *set,
	const char *str)
{
	isl_ctx *ctx;
	isl_set *set2;

	if (!str)
		return set;

	ctx = isl_set_get_ctx(set);
	set2 = isl_set_read_from_str(ctx, str);
	set = isl_set_intersect(set, set2);

	return set;
}

static int getNumberOfScopStmts(polly::Scop *S) {
  int N = 0;
  for (ScopStmt *Stmt : *S)
    N++;

  return N;
}

static ScopStmt **extractScopStmts(polly::Scop *S) {
  int n = getNumberOfScopStmts(S);
  ScopStmt **Res = (ScopStmt **)malloc(n * sizeof(ScopStmt *));

  int K = 0;
  for (ScopStmt *Stmt : *S) {
    Res[K] = Stmt;
    K++;
  }

  return Res;
}

/* Get the number of the arrays by counting different base address of the
 * memory accesses.
 */
static int getNumberOfArrays(polly::Scop *S,
                             std::set<const llvm::Value *> &Bases) {
  for (ScopStmt *Stmt : *S) {
    for (MemoryAccess *Acc : *Stmt) {
      const Value *BaseAddr = Acc->getBaseAddr();
      if (Bases.find(BaseAddr) == Bases.end())
        Bases.insert(BaseAddr);
    }
  }

  return Bases.size();
}

/* Set the size of index "pos" of "array" to "size".
 * In particular, add a constraint of the form
 *
 *	i_pos < size
 *
 * to array->extent and a constraint of the form
 *
 *	size >= 0
 *
 * to array->context.
 */
static struct pet_array *update_size(struct pet_array *array, int pos,
                                     __isl_take isl_pw_aff *size) {
  isl_set *valid;
  isl_set *univ;
  isl_set *bound;
  isl_space *dim;
  isl_aff *aff;
  isl_pw_aff *index;
  isl_id *id;

  valid = isl_pw_aff_nonneg_set(isl_pw_aff_copy(size));
  array->context = isl_set_intersect(array->context, valid);

  dim = isl_set_get_space(array->extent);
  aff = isl_aff_zero_on_domain(isl_local_space_from_space(dim));
  aff = isl_aff_add_coefficient_si(aff, isl_dim_in, pos, 1);
  univ = isl_set_universe(isl_aff_get_domain_space(aff));
  index = isl_pw_aff_alloc(univ, aff);

  size = isl_pw_aff_add_dims(size, isl_dim_in,
                             isl_set_dim(array->extent, isl_dim_set));
  id = isl_set_get_tuple_id(array->extent);
  size = isl_pw_aff_set_tuple_id(size, isl_dim_in, id);
  bound = isl_pw_aff_lt_set(index, size);

  array->extent = isl_set_intersect(array->extent, bound);

  if (!array->context || !array->extent) {
    free(array);
    return NULL;
  }

  return array;
}

/* Extract an integer from "val", which assumed to be non-negative.
 */
static __isl_give isl_val *extract_unsigned(isl_ctx *ctx,
                                            const llvm::APInt &val) {
  unsigned n;
  const uint64_t *data;

  data = val.getRawData();
  n = val.getNumWords();
  return isl_val_int_from_chunks(ctx, n, sizeof(uint64_t), data);
}

/* Extract an affine expression from the APInt "val", which is assumed
 * to be non-negative.
 */
static __isl_give isl_pw_aff *extract_affine(isl_ctx *ctx,
                                             const llvm::APInt &val) {
  isl_space *dim = isl_space_params_alloc(ctx, 0);
  isl_local_space *ls = isl_local_space_from_space(isl_space_copy(dim));
  isl_aff *aff = isl_aff_zero_on_domain(ls);
  isl_set *dom = isl_set_universe(dim);
  isl_val *v;

  v = extract_unsigned(ctx, val);
  aff = isl_aff_add_constant_val(aff, v);

  return isl_pw_aff_alloc(dom, aff);
}

/* Figure out the size of the array at position "pos" and all
 * subsequent positions from "type" and update "array" accordingly.
 */
static struct pet_array *set_upper_bounds(isl_ctx *ctx, struct pet_array *array,
                                          const Type *type, int pos) {
  const ArrayType *atype;
  isl_pw_aff *size;

  if (!array)
    return NULL;

  if (atype = dyn_cast<ArrayType>(type)) {
    size = extract_affine(ctx, llvm::APInt(64, atype->getNumElements()));
    array = update_size(array, pos, size);
    type = atype->getElementType();
    return set_upper_bounds(ctx, array, type, pos /*+ 1, update only 0*/);
  } else
    return array;
}

/* Extract or restore array information from Scop.
 */
static struct pet_array **
extractArraysInfo(polly::Scop *S, std::set<const llvm::Value *> ArrayBases) {
  unsigned N = ArrayBases.size();
  if (N == 0)
    return NULL;

  struct pet_array **Arrays =
      (struct pet_array **)malloc(N * sizeof(struct pet_array *));

  if (!Arrays)
    return NULL;

  isl_ctx *Ctx = S->getIslCtx();

  int J = 0;
  for (const Value *BaseAddr : ArrayBases) {
    struct pet_array *Arr =
        (struct pet_array *)malloc(sizeof(struct pet_array));
    if (!Arr)
      return NULL;
    Arrays[J] = Arr;

    const GlobalVariable *GV = dyn_cast<const GlobalVariable>(BaseAddr);
    ArrayType *ATy = dyn_cast<ArrayType>(GV->getType()->getElementType());
    ArrayType *PTy = ATy;
    ArrayType *EleTy;
    while (EleTy = dyn_cast<ArrayType>(PTy->getElementType()))
      PTy = EleTy;

    void *Addr = const_cast<void *>((const void *)BaseAddr);
    std::string AddrName = "MemRef_" + BaseAddr->getName().str();
    isl_space *Dim = isl_space_set_alloc(Ctx, 0, 1 /*Depth, linearized*/);
    Dim = isl_space_set_tuple_name(Dim, isl_dim_set, AddrName.c_str());
    Arrays[J]->extent = isl_set_nat_universe(Dim);

    Dim = isl_space_params_alloc(Ctx, 0);
    Arrays[J]->context = isl_set_universe(Dim);

    // Arrays[J] = set_upper_bounds(Ctx, Arrays[J], ATy, 0);
    Arrays[J]->value_bounds = NULL;

    std::string TypeName;
    raw_string_ostream OS(TypeName);
    ATy->getElementType()->print(OS);
    TypeName = OS.str();
    Arrays[J]->element_type = (char *)TypeName.c_str();
    Arrays[J]->element_is_record = 0;
    Arrays[J]->element_size =
        ATy->getElementType()->getPrimitiveSizeInBits() / 8;
    Arrays[J]->live_out = 0;
    Arrays[J]->uniquely_defined = 0;
    Arrays[J]->declared = 0;
    Arrays[J]->exposed = 0;
    J++;
  }

  return Arrays;
}

void *ppcg_scop_free(struct ppcg_scop *ps)
{
	if (!ps)
		return NULL;

	isl_set_free(ps->context);
	isl_union_set_free(ps->domain);
	isl_union_set_free(ps->call);
	isl_union_map_free(ps->tagged_reads);
	isl_union_map_free(ps->reads);
	isl_union_map_free(ps->live_in);
	isl_union_map_free(ps->tagged_may_writes);
	isl_union_map_free(ps->tagged_must_writes);
	isl_union_map_free(ps->may_writes);
	isl_union_map_free(ps->must_writes);
	isl_union_map_free(ps->live_out);
	isl_union_map_free(ps->tagged_must_kills);
	isl_union_map_free(ps->tagged_dep_flow);
	isl_union_map_free(ps->dep_flow);
	isl_union_map_free(ps->dep_false);
	isl_union_map_free(ps->dep_external);
	isl_union_map_free(ps->tagged_dep_order);
	isl_union_map_free(ps->dep_order);
	isl_union_map_free(ps->schedule);
	isl_union_map_free(ps->tagger);
	isl_union_map_free(ps->independence);

	free(ps);

	return NULL;
}

/* Tag the access relation "access" with "id".
 * That is, insert the id as the range of a wrapped relation
 * in the domain of "access".
 *
 * If "access" is of the form
 *
 *	D[i] -> A[a]
 *
 * then the result is of the form
 *
 *	[D[i] -> id[]] -> A[a]
 */
__isl_give isl_map *tag_access(__isl_take isl_map *access,
                               __isl_take isl_id *id) {
  isl_space *space;
  isl_map *add_tag;

  space = isl_space_range(isl_map_get_space(access));
  space = isl_space_from_range(space);
  space = isl_space_set_tuple_id(space, isl_dim_in, id);
  add_tag = isl_map_universe(space);
  access = isl_map_domain_product(access, add_tag);

  return access;
}

static int NumRef = 0;

/* Collect and return all read access relations (if "read" is set)
 * and/or all write access relations (if "write" is set) in "stmt".
 * If "tag" is set, then the access relations are tagged with
 * the corresponding reference identifiers.
 * If "kill" is set, then "stmt" is a kill statement and we simply
 * add the argument of the kill operation.
 *
 * If "must" is set, then we only add the accesses that are definitely
 * performed.  Otherwise, we add all potential accesses.
 * In particular, if the statement has any arguments, then if "must" is
 * set we currently skip the statement completely.  If "must" is not set,
 * we project out the values of the statement arguments.
 */
static __isl_give isl_union_map *
stmt_collect_accesses(ScopStmt *stmt, int read, int write, int kill, int must,
                      int tag, __isl_take isl_space *dim) {
  isl_union_map *accesses;
  isl_set *domain;

  if (!stmt)
    return NULL;

  accesses = isl_union_map_empty(dim);

  if (must && stmt->/*n_arg*/ getNumParams() > 0)
    return accesses;

  domain = /*isl_set_copy(stmt->domain)*/ stmt->getDomain();
  if (isl_set_is_wrapping(domain))
    domain = isl_map_domain(isl_set_unwrap(domain));

  /* comment out by Yabin Hu
    if (kill)
      accesses = expr_collect_access(stmt->body->args[0], tag, accesses,
    domain);
    else
      accesses = expr_collect_accesses(stmt->body, read, write, must, tag,
                                       accesses, domain);
  */
  isl_map *Access = nullptr;

  for (MemoryAccess *Acc : *stmt) {
    if ((read && Acc->isRead()) || (write && must && Acc->isMustWrite()) ||
        (write && (must == 0) && Acc->isMayWrite())) {
      isl_set *Domain = isl_set_copy(domain);
      Access = Acc->getAccessRelation();
      Access = isl_map_intersect_domain(Access, Domain);
      if (tag) {
        Access = tag_access(Access, Acc->getRefId());
      }
      accesses = isl_union_map_add_map(accesses, Access);
    }
  }

  isl_set_free(domain);

  return accesses;
}

/* Compute a mapping from all outer arrays (of structs) in scop
 * to their innermost arrays.
 *
 * In particular, for each array of a primitive type, the result
 * contains the identity mapping on that array.
 * For each array involving member accesses, the result
 * contains a mapping from the elements of the outer array of structs
 * to all corresponding elements of the innermost nested arrays.
 */
static __isl_give isl_union_map *compute_to_inner(struct ppcg_scop *scop)
{
	int i;
	isl_union_map *to_inner;

	to_inner = isl_union_map_empty(isl_set_get_space(scop->context));

	for (i = 0; i < scop->n_array; ++i) {
		struct pet_array *array = scop->arrays[i];
		isl_set *set;
		isl_map *map;

		if (array->element_is_record)
			continue;

		set = isl_set_copy(array->extent);
		map = isl_set_identity(isl_set_copy(set));

		while (set && isl_set_is_wrapping(set)) {
			isl_id *id;
			isl_map *wrapped;

			id = isl_set_get_tuple_id(set);
			wrapped = isl_set_unwrap(set);
			wrapped = isl_map_domain_map(wrapped);
			wrapped = isl_map_set_tuple_id(wrapped, isl_dim_in, id);
			map = isl_map_apply_domain(map, wrapped);
			set = isl_map_domain(isl_map_copy(map));
		}

		map = isl_map_gist_domain(map, set);

		to_inner = isl_union_map_add_map(to_inner, map);
	}

	return to_inner;
}

/* Collect and return all read access relations (if "read" is set)
 * and/or all write access relations (if "write" is set) in "scop".
 * If "kill" is set, then we only add the arguments of kill operations.
 * If "must" is set, then we only add the accesses that are definitely
 * performed.  Otherwise, we add all potential accesses.
 * If "tag" is set, then the access relations are tagged with
 * the corresponding reference identifiers.
 * For accesses to structures, the returned access relation accesses
 * all individual fields in the structures.
 */
static __isl_give isl_union_map *scop_collect_accesses(struct ppcg_scop *scop,
                                                       int read, int write,
                                                       int kill, int must,
                                                       int tag) {
  int i;
  isl_union_map *accesses;
  isl_union_set *arrays;
  isl_union_map *to_inner;

  if (!scop)
    return NULL;

  accesses = isl_union_map_empty(isl_set_get_space(scop->context));

  for (i = 0; i < scop->n_stmt; ++i) {
    ScopStmt *stmt = scop->stmts[i];
    isl_union_map *accesses_i;
    isl_space *space;

    // Comment by Yabin Hu
    // If kill is set, we collect nothing.
    if (kill /*&& !is_kill(stmt)*/)
      continue;

    space = isl_set_get_space(scop->context);
    accesses_i =
        stmt_collect_accesses(stmt, read, write, kill, must, tag, space);
    accesses = isl_union_map_union(accesses, accesses_i);
  }

  arrays = isl_union_set_empty(isl_union_map_get_space(accesses));
  for (i = 0; i < scop->n_array; ++i) {
    isl_set *extent = isl_set_copy(scop->arrays[i]->extent);
    arrays = isl_union_set_add_set(arrays, extent);
  }
  accesses = isl_union_map_intersect_range(accesses, arrays);

  to_inner = compute_to_inner(scop);
  accesses = isl_union_map_apply_range(accesses, to_inner);

  return accesses;
}

/* Collect all potential read access relations.
 */
static __isl_give isl_union_map *
pet_scop_collect_may_reads(struct ppcg_scop *scop) {
  return scop_collect_accesses(scop, 1, 0, 0, 0, 0);
}

/* Collect all potential write access relations.
 */
static __isl_give isl_union_map *
pet_scop_collect_may_writes(struct ppcg_scop *scop) {
  return scop_collect_accesses(scop, 0, 1, 0, 0, 0);
}

/* Collect all definite write access relations.
 */
static __isl_give isl_union_map *
pet_scop_collect_must_writes(struct ppcg_scop *scop) {
  return scop_collect_accesses(scop, 0, 1, 0, 1, 0);
}

/* Collect all definite kill access relations.
 */
static __isl_give isl_union_map *
pet_scop_collect_must_kills(struct ppcg_scop *scop) {
  return scop_collect_accesses(scop, 0, 0, 1, 1, 0);
}

/* Collect all tagged potential read access relations.
 */
static __isl_give isl_union_map *
pet_scop_collect_tagged_may_reads(struct ppcg_scop *scop) {
  return scop_collect_accesses(scop, 1, 0, 0, 0, 1);
}

/* Collect all tagged potential write access relations.
 */
static __isl_give isl_union_map *
pet_scop_collect_tagged_may_writes(struct ppcg_scop *scop) {
  return scop_collect_accesses(scop, 0, 1, 0, 0, 1);
}

/* Collect all tagged definite write access relations.
 */
static __isl_give isl_union_map *
pet_scop_collect_tagged_must_writes(struct ppcg_scop *scop) {
  return scop_collect_accesses(scop, 0, 1, 0, 1, 1);
}

/* Collect all tagged definite kill access relations.
 */
__isl_give isl_union_map *
pet_scop_collect_tagged_must_kills(struct ppcg_scop *scop) {
  return scop_collect_accesses(scop, 0, 0, 1, 1, 1);
}

static __isl_give isl_union_map *collectSchedule(Scop *S) {
  isl_union_map *Schedule;

  if (!S)
    return NULL;

  Schedule = isl_union_map_empty(S->getParamSpace());

  for (ScopStmt *Stmt : *S) {
    isl_map *StmtSchedule = Stmt->getScattering();

    StmtSchedule = isl_map_intersect_domain(StmtSchedule, Stmt->getDomain());
    Schedule =
        isl_union_map_union(Schedule, isl_union_map_from_map(StmtSchedule));
  }

  return Schedule;
}

/* Extract a ppcg_scop from a pet_scop.
 *
 * The constructed ppcg_scop refers to elements from the pet_scop
 * so the pet_scop should not be freed before the ppcg_scop.
 */
struct ppcg_scop *
ppcg_scop_from_polly_scop(Scop *scop, struct ppcg_options *options)
{
	int i;
	isl_ctx *ctx;
	struct ppcg_scop *ps;

	if (!scop)
		return NULL;

	isl_set *Context = scop->getContext();
        ctx = isl_set_get_ctx(Context);

	ps = isl_calloc_type(ctx, struct ppcg_scop);
	if (!ps)
		return NULL;

	ps->options = options;
	ps->start = /*pet_loc_get_start(scop->loc)*/0;
	ps->end = /*pet_loc_get_end(scop->loc)*/0;
	ps->context = /*isl_set_copy(*/Context;
	ps->context = set_intersect_str(ps->context, options->ctx);
	ps->domain = /*collect_non_kill_domains(*/scop->getDomains();
	ps->call = collect_call_domains(scop);
	std::set<const llvm::Value *> ArrayBases;
	ps->n_array = getNumberOfArrays(scop, ArrayBases);
	ps->arrays = extractArraysInfo(scop, ArrayBases);
	ps->n_stmt = getNumberOfScopStmts(scop);
	ps->stmts = extractScopStmts(scop);
	ps->tagged_reads = pet_scop_collect_tagged_may_reads(ps);
	ps->reads = pet_scop_collect_may_reads(ps);
	ps->tagged_may_writes = pet_scop_collect_tagged_may_writes(ps);
	ps->may_writes = pet_scop_collect_may_writes(ps);
	ps->tagged_must_writes = pet_scop_collect_tagged_must_writes(ps);
	ps->must_writes = pet_scop_collect_must_writes(ps);
	ps->tagged_must_kills = pet_scop_collect_tagged_must_kills(ps);
	ps->may_writes =
            isl_union_map_union(ps->may_writes,
                                isl_union_map_copy(ps->must_writes));
	ps->tagged_may_writes =
            isl_union_map_union(ps->tagged_may_writes,
                                isl_union_map_copy(ps->tagged_must_writes));
	ps->schedule = collectSchedule(scop);
	// ps->n_type = scop->n_type;
	// ps->types = scop->types;
#if 0
        ps->n_independence = scop->n_independence;
	ps->independences = scop->independences;
	ps->independence = isl_union_map_empty(isl_set_get_space(ps->context));
	for (i = 0; i < ps->n_independence; ++i)
		ps->independence = isl_union_map_union(ps->independence,
			isl_union_map_copy(ps->independences[i]->filter));
#endif

	compute_tagger(ps);
	compute_dependences(ps);
	eliminate_dead_code(ps);

	if (!ps->context || !ps->domain || !ps->call || !ps->reads ||
	    !ps->may_writes || !ps->must_writes || !ps->tagged_must_kills ||
	    !ps->schedule || !ps->independence)
		return (struct ppcg_scop *)ppcg_scop_free(ps);

	return ps;
}

/* Internal data structure for ppcg_transform.
 */
struct ppcg_transform_data {
	struct ppcg_options *options;
	__isl_give isl_printer *(*transform)(__isl_take isl_printer *p,
		struct ppcg_scop *scop, void *user);
	void *user;
};

/* Callback for pet_transform_C_source that transforms
 * the given pet_scop to a ppcg_scop before calling the
 * ppcg_transform callback.
 *
 * If "scop" contains any data dependent conditions or if we may
 * not be able to print the transformed program, then just print
 * the original code.
 */
static __isl_give isl_printer *transform(__isl_take isl_printer *p,
	struct pet_scop *scop, void *user)
{
	struct ppcg_transform_data *data = (struct ppcg_transform_data *)user;
	struct ppcg_scop *ps;

	if (!pet_scop_can_build_ast_exprs(scop) ||
	    pet_scop_has_data_dependent_conditions(scop)) {
		p = pet_scop_print_original(scop, p);
		pet_scop_free(scop);
		return p;
	}

	scop = pet_scop_align_params(scop);
	// ps = ppcg_scop_from_pet_scop(scop, data->options);

	p = data->transform(p, ps, data->user);

	ppcg_scop_free(ps);
	pet_scop_free(scop);

	return p;
}

/* Transform the C source file "input" by rewriting each scop
 * through a call to "transform".
 * The transformed C code is written to "out".
 *
 * This is a wrapper around pet_transform_C_source that transforms
 * the pet_scop to a ppcg_scop before calling "fn".
 */
int ppcg_transform(isl_ctx *ctx, const char *input, FILE *out,
	struct ppcg_options *options,
	__isl_give isl_printer *(*fn)(__isl_take isl_printer *p,
		struct ppcg_scop *scop, void *user), void *user)
{
	struct ppcg_transform_data data = { options, fn, user };
	return pet_transform_C_source(ctx, input, out, &transform, &data);
}

/* Check consistency of options.
 *
 * Return -1 on error.
 */
#if 0
static int check_options(isl_ctx *ctx)
{
	struct options *options;

	options = (struct options *)isl_ctx_peek_options(ctx, &options_args);
	if (!options)
		isl_die(ctx, isl_error_internal,
			"unable to find options", return -1);

	if (options->ppcg->openmp &&
	    !isl_options_get_ast_build_atomic_upper_bound(ctx))
		isl_die(ctx, isl_error_invalid,
			"OpenMP requires atomic bounds", return -1);

	return 0;
}

int main(int argc, char **argv)
{
	int r;
	isl_ctx *ctx;
	struct options *options;

	options = options_new_with_defaults();
	assert(options);

	ctx = isl_ctx_alloc_with_options(&options_args, options);
	isl_options_set_schedule_outer_coincidence(ctx, 1);
	isl_options_set_schedule_maximize_band_depth(ctx, 1);
	pet_options_set_encapsulate_dynamic_control(ctx, 1);
	argc = options_parse(options, argc, argv, ISL_ARG_ALL);

	if (check_options(ctx) < 0)
		r = EXIT_FAILURE;
	else if (options->ppcg->target == PPCG_TARGET_CUDA)
		r = generate_cuda(ctx, options->ppcg, options->input);
	else if (options->ppcg->target == PPCG_TARGET_OPENCL)
		r = generate_opencl(ctx, options->ppcg, options->input,
				options->output);
	else
		r = generate_cpu(ctx, options->ppcg, options->input,
				options->output);

	isl_ctx_free(ctx);

	return r;
}
#endif
