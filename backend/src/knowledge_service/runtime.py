from langgraph.checkpoint.mongodb import MongoDBSaver

from knowledge_service.adapters.mineru import MinerU
from knowledge_service.adapters.models import Models
from knowledge_service.adapters.storage import Objects, Relations, Vectors
from knowledge_service.repositories.mongo import MongoStore
from knowledge_service.services import Domain


class Runtime:
    def __init__(self, settings):
        self.s = settings
        self.store = MongoStore(settings)
        self.domain = Domain(self.store, settings)
        self.models = Models(settings)
        self.mineru = MinerU(settings)
        self._objects = self._vectors = self._relations = self._checkpointer = None

    @property
    def objects(self):
        if self._objects is None:
            self._objects = Objects(self.s)
        return self._objects

    @property
    def vectors(self):
        if self._vectors is None:
            self._vectors = Vectors(self.s)
        return self._vectors

    @property
    def relations(self):
        if self._relations is None:
            self._relations = Relations(self.s)
        return self._relations

    @property
    def checkpointer(self):
        if self._checkpointer is None:
            self._checkpointer = MongoDBSaver(self.store.client, db_name=self.s.graph_checkpoint_database)
        return self._checkpointer

    async def close(self):
        await self.models.close()
        await self.mineru.close()
        for resource in (self._vectors, self._relations):
            if resource:
                resource.close()
        self.store.close()
