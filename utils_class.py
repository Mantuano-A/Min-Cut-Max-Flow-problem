class Relation:
    """
    This class keeps track of each edge present between the source and the target and contains all the information 
    necessary to recognize the relation
    Attributes:
        source: (classe User) indicate node source
        target: (classe User) indicate node target
        weight: (int) weight of edge
    """
    def __init__(self, source, target, weight):
        self.source_ = source.get_ID
        self.target_ = target.get_ID
        self.weight_ = weight
        self.weight_in_ = 0
    
    @property
    def target(self):
        return self.target_
    
    @property
    def source(self):
        return self.source_
    
    @property
    def weight(self):
        return self.weight_
    
    @property
    def weight_in(self):
        return self.weight_in_
    
    def set_weight_in(self, weight):
        self.weight_in_ = weight
    
    def set_weight(self, weight):
        self.weight_ = weight
        
    def __str__(self): 
        return "{\"source\": " + str(self.source_) + ", \"target\": " + \
        str(self.target_) + ", \"weight\": "+ str(self.weight_) + "}"
    
    def __repr__(self): 
        return self.__str__()
    
class User:
    """
    this class contains information about the user and his relations
    Attributes:
        ID_user: (int) ID of user
    """
    def __init__(self, ID_user):
        self.ID_user = ID_user
        self.in_relation = dict()
        self.out_relation = dict()
    
    def add_in_relation(self, in_relation):
        """
        Add relation in User with incoming edge
        
            Parameters:
                    in_relation (Relation) relation between self and v
        """

        self.in_relation[in_relation.source] = (in_relation)

    
    def add_out_relation(self, out_relation):
        """
        Add relation in User with outgoing edge
        
            Parameters:
                    in_relation (Relation) relation between self and u
        """

        self.out_relation[out_relation.target] = (out_relation)
    
    def set_in_relation(self, inRelations):
        self.in_relation = inRelations
    
    def set_out_relation(self, outRelation):
        self.out_relation = outRelation
    
    @property
    def get_ID(self):
        return self.ID_user

    @property
    def get_in_relation(self):
        return self.in_relation
    
    @property
    def get_out_relation(self):
        return self.out_relation

    def __str__(self):
        return "{\"in_relation\": " + str(self.in_relation) +  ", \"out_relation\": " + str(self.out_relation) + "}"

    def __repr__(self): 
        return self.__str__()