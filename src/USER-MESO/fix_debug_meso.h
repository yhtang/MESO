//#ifndef LMP_MESO_FIX_DEBUG
//#define LMP_MESO_FIX_DEBUG
//
//#include "fix.h"
//
//namespace LAMMPS_NS {
//
//#define N_SLOT 1000
//
//class MesoFixDebug : public Fix
//{
//public:
//  MesoFixDebug(class LAMMPS *, int, char **);
//  ~MesoFixDebug();
//  virtual void init();
//  virtual int setmask();
//  virtual void initial_integrate(int);
//  virtual void post_integrate();
//  virtual void pre_exchange();
//  virtual void pre_neighbor();
//  virtual void pre_force(int);
//  virtual void post_force(int);
//  virtual void final_integrate();
//  virtual void end_of_step();
//
//protected:
//  virtual void vtotal( string tag );
//  virtual void dump();
//  int monitor_target;
//
//  int p;
//  r32* Total;
//  ofstream fout;
//  vector<string> Tags;
//  vector<int> nT;
//};
//
//}
//
//#endif
